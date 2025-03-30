# Speech-to-Speech Conversational AI

A real-time conversational AI system that enables natural speech-to-speech interactions using state-of-the-art models for speech recognition, language modeling, and text-to-speech synthesis.

## Overview

This application provides an end-to-end conversational experience by:

1. Capturing audio input from the user's microphone
2. Using Silero VAD for voice activity detection
3. Transcribing speech to text using Whisper Large V3
4. Processing text with Llama 3.1 8B Instruct model for intelligent responses
5. Converting responses to natural speech using CosyVoice TTS
6. Playing back audio responses with interruption handling

## Features

- Real-time speech recognition and response generation
- Interruption handling for natural conversation flow
- Expressive speech synthesis with emotion cues
- Low-latency first response time
- Streaming audio generation for responsive interactions

## Prerequisites

- CUDA-capable GPU (for optimal performance)
- Docker (for running the LLM service)
- Conda (for environment management)
- Git LFS (for downloading large model files)

## Installation

### 1. Setup Environment

```bash
conda create --name sts python=3.9
conda activate sts
pip install silero-vad==5.1.2
pip install insanely-fast-whisper==0.0.15
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

### 2. Setup LLM via NVIDIA NeMo Inference Microservice (NIM)

```bash
docker login nvcr.io
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export LOCAL_NIM_CACHE=~/.cache/nim
docker run -it --rm \
    --gpus '"device=0"' \
    --shm-size=16GB \
    -e NGC_API_KEY -e NIM_RELAX_MEM_CONSTRAINTS=1 -e NIM_NUM_KV_CACHE_SEQ_LENS=1 \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

You may use other LLM inference services such as vLLM or SGLang as well.

### 3. Download Required Models

```bash
git lfs install
git clone https://huggingface.co/openai/whisper-large-v3 models/whisper-large-v3
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git models/CosyVoice-300M-Instruct
```

### 4. Set up Matcha-TTS (Required for CosyVoice)

The project uses Matcha-TTS as a dependency for the CosyVoice TTS system.

## Usage

### Start the Application

```bash
export PYTHONPATH=third_party/Matcha-TTS
python app.py
```

The application will start listening for voice input through your microphone. Speak naturally, and the system will respond with synthesized speech.

## Configuration

You can modify the following parameters in `app.py`:

- `LLM_ENDPOINT`: The endpoint for the LLM service (default: 'http://localhost:8000/v1/chat/completions')
- `LLM_MODEL`: The model to use for language processing (default: "meta/llama-3.1-8b-instruct")
- `LLM_MAX_TOKENS`: Maximum tokens for LLM response (default: 1000)
- `TRANSCRIBE_MODEL_PATH`: Path to the Whisper model (default: "models/whisper-large-v3")
- `TTS_MODEL_PATH`: Path to the CosyVoice model (default: "models/CosyVoice-300M-Instruct")

## System Requirements

- CUDA-compatible GPU (recommended)
- At least 16GB of GPU memory for optimal performance
- Python 3.9
- Sufficient disk space for model storage (approximately 10GB)