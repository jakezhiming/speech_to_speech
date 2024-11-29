# Setup LLM
```
docker login nvcr.io
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export LOCAL_NIM_CACHE=~/.cache/nim
docker run -it --rm     --gpus '"device=6"'     --shm-size=16GB     -e NGC_API_KEY -e NIM_RELAX_MEM_CONSTRAINTS=1 -e NIM_NUM_KV_CACHE_SEQ_LENS=1 -v "$LOCAL_NIM_CACHE:/opt/nim/.cache"     -u $(id -u)     -p 30001:8000   nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```
# Setup Environment
```
conda create --name sts python=3.9
conda activate sts
pip install silero-vad==5.1.2
pip install insanely-fast-whisper==0.0.15
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

# Download Models
```
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
```

# Start App
```
export PYTHONPATH=third_party/Matcha-TTS
python app.py
```