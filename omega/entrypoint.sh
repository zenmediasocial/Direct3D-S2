#!/bin/bash

echo "[Ω] Omega Brick Dev Container Booting..."

# -------- Ensure Model Directory Exists --------
mkdir -p /models
cd /models

# -------- Download Whisper base.en (if needed) --------
if [ ! -f "whisper-base.en.pt" ]; then
  echo "[Ω] Downloading Whisper base.en model..."
  curl -L -o whisper-base.en.pt https://huggingface.co/openai/whisper/resolve/main/whisper-base.en.pt
else
  echo "[Ω] Whisper model already present."
fi

# -------- Download & Cache Qwen-Tiny (transformers) --------
echo "[Ω] Checking Qwen model cache..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-0.5B')
AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B')
"

# -------- Launch Flask App --------
if [ -f /flask_app/app.py ]; then
  echo "[Ω] Starting Flask app..."
  cd /flask_app
  export FLASK_APP=app.py
  export FLASK_ENV=development
  exec flask run --host=0.0.0.0 --port=5000
else
  echo "[Ω] ERROR: No /flask_app/app.py found. Mount or copy your Flask app."
  exec /bin/bash
fi

