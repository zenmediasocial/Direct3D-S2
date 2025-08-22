from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import whisper

app = Flask(__name__)

print("[Ω] Loading models...")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")
qwen_pipe = pipeline("text-generation", model=qwen_model, tokenizer=qwen_tokenizer)

whisper_model = whisper.load_model("base.en", download_root="/models")

@app.route("/ask", methods=["POST"])
def ask():
    prompt = request.json.get("prompt")
    result = qwen_pipe(prompt, max_new_tokens=100)
    return jsonify({"response": result[0]["generated_text"]})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_path = request.json.get("path")
    result = whisper_model.transcribe(audio_path)
    return jsonify(result)

