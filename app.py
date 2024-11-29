import os
import re
import time
import math
import torch
import queue
import requests
import json
import shutil
import threading
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from transformers import pipeline
from silero_vad import load_silero_vad, get_speech_timestamps

from cosyvoice.cli.cosyvoice import CosyVoice

llm_url = 'http://172.16.17.2:30001/v1/chat/completions'
llm_headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
llm_payload = {
    "model": "meta/llama-3.1-8b-instruct",
    "max_tokens": 1000,
    "stream": True
}

audio_temp_folder = "audio_tmp"
if os.path.exists(audio_temp_folder):
    shutil.rmtree(audio_temp_folder)
os.makedirs(audio_temp_folder, exist_ok=True)

logs_folder = "logs"
if os.path.exists(logs_folder):
    shutil.rmtree(logs_folder)
os.makedirs(logs_folder, exist_ok=True)

listening_interval = 0.3
sampling_rate = 16000 

vad_model = load_silero_vad()

transcribe_pipe = pipeline(
    "automatic-speech-recognition",
    model="/home/jake/models/whisper-large-v3",
    torch_dtype=torch.float16,
    device="cuda:0",
    model_kwargs={"attn_implementation": "sdpa"},
)

tts_queue = queue.Queue()
chunk_intervals = [5, 25, 125, 625]

cosyvoice = CosyVoice('/home/jake/models/CosyVoice-300M-Instruct', load_jit=False, load_onnx=False, fp16=True)

rounds = 0
first_audio = True
interrupt = False
interrupt_threshold = 3

def listening_callback(indata, frames, time, status):
    if status:
        print(status)
    global audio_buffer
    audio_buffer = np.append(audio_buffer, indata[:, 0])

def tts_worker(tts_queue, audio_temp_folder, interruption_event):
    while True:
        item = tts_queue.get()
        if item is None:
            break
        sequence_num, partial_reply = item
        cleaned_text = re.sub(r'[\*#]', '', partial_reply)

        output_data_list = []
        for output_data in cosyvoice.inference_instruct(
            cleaned_text,
            '英文女',
            'Assistant is a young female speaker.',
            stream=True
        ):
            if interruption_event.is_set():
                print("Inference interrupted!")
                while not tts_queue.empty():
                    try:
                        tts_queue.get_nowait()
                        tts_queue.task_done()
                    except queue.Empty:
                        break 
                break
            audio_tensor = output_data['tts_speech']
            audio_data = audio_tensor.numpy().T
            output_data_list.append(audio_data)

        if len(output_data_list) > 0:
            combined_audio_data = np.concatenate(output_data_list, axis=0)
            filename = f"audio_{sequence_num:04d}.wav"
            wavfile.write(os.path.join(audio_temp_folder, filename), 22050, combined_audio_data)

        tts_queue.task_done()

def delete_all_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def audio_playback(audio_temp_folder, interruption_event):
    global first_audio, interrupt

    while True:
        if interrupt:
            delete_all_files(audio_temp_folder)
            interruption_event.set()
            interrupt = False
            continue

        files = sorted(
            [f for f in os.listdir(audio_temp_folder) if f.endswith('.wav')],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )

        for file in files:
            file_path = os.path.join(audio_temp_folder, file)
            sampling_rate, audio_data = wavfile.read(file_path)
            if first_audio:
                print(f"{'*' * 50}\nTime to first audio: {time.time() - latency:.2f} seconds\n{'*' * 50}")
                first_audio = False

            if play_audio(audio_data, sampling_rate):
                os.remove(file_path)
        time.sleep(0.1)

def play_audio(audio_data, sampling_rate):
    global interrupt
    if interrupt:
        return False
    sd.play(audio_data, sampling_rate)
    while sd.get_stream().active:
        if interrupt:
            sd.stop()
            return False
        time.sleep(0.1)
    return True

interruption_event = threading.Event()

tts_thread = threading.Thread(target=tts_worker, args=(tts_queue, audio_temp_folder, interruption_event))
tts_thread.daemon = True
tts_thread.start()

playback_thread = threading.Thread(target=audio_playback, args=(audio_temp_folder,interruption_event))
playback_thread.daemon = True
playback_thread.start()

system_prompt = """You are a Conversational Assistant. Your role is to craft articulate and engaging responses that will enhance the user's auditory experience. Follow these guidelines to ensure effective and enjoyable interactions:
Focus on Clarity and Brevity: Strive to provide responses that are concise and to the point. Expand only when the user explicitly requests more information.
Incorporate Expressive Cues: Use <laughter> ... </laughter> to indicate words that invite humor, and <strong> ... </strong> to emphasize certain words. Employ [laughter] to simulate genuine laughter in your response, and use [breath] to suggest pauses for natural pacing. Only <laughter></laughter>, <strong></strong>, [laughter], and [breath] are allowed. Do not use any other cues other than the ones mentioned.
Use Verbal-friendly Content: Avoid special characters, code, and math equations, as these elements are not suited for verbal communication.
Enhance Engagement and Accessibility: Ensure that your responses are not only informative but also engaging and easy to understand, maintaining the listener's interest.
Be Attuned to User Indications: Adapt to the user's cues. If more detailed information is requested ("Can you elaborate?"), provide it succinctly yet comprehensively.
Maintain a Warm and Inviting Tone: Adopt a friendly and approachable tone, creating a pleasant and enjoyable auditory experience for the user.
"""
messages = [{"role": "system", "content": system_prompt}]

print("Started!")
while True:
    audio_buffer = np.empty((0,), dtype=np.float32)
    stream = sd.InputStream(callback=listening_callback, channels=1, samplerate=sampling_rate)
    stream.start()

    try:
        start = 0
        prev_end = 0
        prev_blocks_num = 0
        loops = 0
        interrupt_count = 0

        while True:
            time.sleep(listening_interval)
            audio_max = np.max(np.abs(audio_buffer))
            if audio_max > 0:
                audio_np = (audio_buffer / audio_max * 32767).astype(np.int16)
            else:
                audio_np = (audio_buffer * 32767).astype(np.int16)

            audio_tensor = torch.from_numpy(audio_np)
            speech_blocks = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sampling_rate, threshold=0.7, min_silence_duration_ms=500)
            
            if len(speech_blocks) == 0 and (loops == 20 or prev_blocks_num > 0):
                audio_buffer = np.empty((0,), dtype=np.float32)
                loops = 0
                prev_blocks_num = 0
                print("Clear audio buffer")
            
            if len(speech_blocks) > 0:
                print(f"Audio detected - start {start} - prev_end {prev_end} - {speech_blocks}")
                interrupt_count += 1
                if interrupt_count >= interrupt_threshold:
                    interrupt = True
                if len(speech_blocks) == 1:
                    start = speech_blocks[0]["start"]
                    end = speech_blocks[0]["end"] 
                    if end == prev_end:
                        break
                    else:
                        prev_end = end

                elif len(speech_blocks) > 1:
                    speech_completed = False
                    for i, block in enumerate(speech_blocks[1:]):
                        block_start = speech_blocks[i+1]["start"]
                        block_end = speech_blocks[i+1]["end"]
                        if len(speech_blocks) <= prev_blocks_num and prev_end == block_end:
                            speech_completed = True
                            break
                        if block_start - prev_end > -sampling_rate:
                            if block_start - prev_end < sampling_rate:
                                prev_end = block_end
                            else:
                                speech_completed = True
                                break
                        else:
                            continue
                    if speech_completed:
                        break

            prev_blocks_num = len(speech_blocks)
            loops += 1

    except KeyboardInterrupt:
        stream.stop()

    finally:
        stream.close()

    latency = time.time()
    start = max(0, start - math.floor(sampling_rate/2))
    prev_end = min(len(audio_np), prev_end + math.floor(sampling_rate/2))
    audio_np = audio_np[start:prev_end]
    rounds += 1
    print(f"{'*' * 50}\nConversation round {rounds} - Start {start} - End {prev_end}\n{'*' * 50}")
    wavfile.write(f"logs/transcribed_audio_{rounds}.wav", sampling_rate, audio_np)
    
    transcribe_outputs = transcribe_pipe(
        audio_np,
        chunk_length_s=30,
        batch_size=24,
        generate_kwargs={"task": "transcribe", "language": "en"},
        return_timestamps=False,
    )
    query = transcribe_outputs['text']
    print(f"\n{'=' * 50}\nTranscribed audio: {query}\n{'=' * 50}\n")

    messages.append({"role": "user", "content": query})

    llm_payload["messages"] = messages
    llm_response = requests.post(llm_url, headers=llm_headers, json=llm_payload, stream=True)

    full_reply = ""
    chunks_collected = []
    chunk_counter = 0
    interval_index = 0
    reply_chunk_num = 0
    interrupt = False
    interruption_event.clear()

    for line in llm_response.iter_lines():
        if line:
            line_content = line.decode('utf-8').strip()
            if line_content == 'data: [DONE]':
                break 
            if line_content.startswith("data: "):
                    data = json.loads(line_content[6:])
                    choice = data['choices'][0]
                    content = choice['delta'].get('content', '')
                    if content:
                        full_reply += content
                        chunks_collected.append(content)
                        chunk_counter += 1
                        partial_reply = "".join(chunks_collected)

                        if interval_index < len(chunk_intervals) and chunk_counter > chunk_intervals[interval_index]:
                            if partial_reply.endswith('\n') or partial_reply.strip().endswith(('.', '?', ':')):
                                chunks_collected = []
                                reply_chunk_num += 1
                                interval_index += 1
                                tts_queue.put((reply_chunk_num, partial_reply))

    if chunks_collected:
        partial_reply = "".join(chunks_collected)
        chunks_collected = []
        reply_chunk_num += 1
        tts_queue.put((reply_chunk_num, partial_reply))

    print(f"\n{'=' * 50}\nReply: {full_reply}\n{'=' * 50}\n")
    messages.append({"role": "assistant", "content": full_reply})