from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import whisper

app = Flask(__name__)

# Load Qwen model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load Whisper model
whisper_model = whisper.load_model("base.en", download_root="/models")

@app.route("/")
def home():
    return "Omega Brick is alive."

@app.route("/ask", methods=["POST"])
def ask():
    prompt = request.json.get("prompt", "")
    result = llm(prompt, max_new_tokens=100)[0]["generated_text"]
    return jsonify({"response": result})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_path = request.json.get("path", "")
    try:
        result = whisper_model.transcribe(audio_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100, debug=True)

