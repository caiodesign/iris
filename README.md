# English Voice Companion

Local, voice-controlled English practice companion. Say "Hey Chat" to start
talking, "Cancel That" to retract what you just said, "Bye Bye" to end the
session.

## Setup

1. Install [Ollama](https://ollama.com) separately (not via pip), then pull the model:
   ```
   ollama pull llama3.1:8b
   ```
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install the NVIDIA libraries faster-whisper needs to run on the GPU
   (plain `pip install faster-whisper` does NOT include them):
   ```
   pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
   ```
   If the app still fails at startup with a CUDA/cuDNN error, open
   `companion/config.py` and set `WHISPER_DEVICE = "cpu"` and
   `WHISPER_COMPUTE_TYPE = "int8"` — slower, but always works.
4. Download a Piper voice (run from the project root, so the file lands here):
   ```
   python -m piper.download_voices en_US-lessac-medium
   ```
5. Run it:
   ```
   python -m companion.main
   ```

## Usage notes

- Say "Cancel That" **in the same breath** as the sentence you want to
  retract (e.g., "I went to... cancel that"). Once you pause, the app has
  already sent what you said to the model and a reply is on its way.
