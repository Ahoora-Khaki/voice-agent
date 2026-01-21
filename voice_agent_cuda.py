import numpy as np
import sounddevice as sd
import webrtcvad
import tempfile
import subprocess
import collections
import time
from scipy.io import wavfile
from faster_whisper import WhisperModel
from llama_cpp import Llama

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SILENCE_FRAMES = 10
VAD_MODE = 3

WHISPER_MODEL = "tiny"
LLM_MODEL = "gemma-2-2b-it-Q4_K_M.gguf"

SYSTEM_PROMPT = "You are a brutally witty voice AI with zero filter but genuine care. Every response flows naturally in one seamless paragraph. Start with a savage roast or joke about their question, then immediately shift to warm helpful problem-solving with practical advice, and end by offering one specific action you can take for them. Keep it under 5 sentences total. No bullet points, no numbering, no formatting. Just natural flowing speech like a brilliant friend who roasts you then helps you. Never say As an AI or break character."

print("Loading models...")
vad = webrtcvad.Vad(VAD_MODE)
whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
llm = Llama(model_path=LLM_MODEL, n_ctx=2048, n_gpu_layers=-1, verbose=False)
conversation_history = []
print("Ready!")

def record():
    print("Listening...")
    audio_buffer = []
    ring_buffer = collections.deque(maxlen=SILENCE_FRAMES)
    speech_started = False
    
    def callback(indata, frames, time_info, status):
        audio_buffer.append(indata.copy())
    
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback, blocksize=FRAME_SIZE)
    stream.start()
    start_time = time.time()
    
    try:
        while time.time() - start_time < 15:
            time.sleep(FRAME_DURATION_MS / 1000)
            if not audio_buffer:
                continue
            frame = audio_buffer[-1].flatten().tobytes()
            try:
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
            except:
                continue
            if is_speech:
                speech_started = True
                ring_buffer.clear()
            elif speech_started:
                ring_buffer.append(1)
                if len(ring_buffer) >= SILENCE_FRAMES:
                    break
    finally:
        stream.stop()
        stream.close()
    
    if not audio_buffer or not speech_started:
        return None
    audio = np.concatenate(audio_buffer).flatten()
    return audio.astype(np.float32) / 32768.0

def transcribe(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, SAMPLE_RATE, (audio * 32768).astype(np.int16))
        segments, _ = whisper.transcribe(f.name)
        return " ".join([s.text for s in segments]).strip()

def speak(text):
    try:
        from kokoro_onnx import Kokoro
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        samples, sr = kokoro.create(text, voice="af_heart", speed=1.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, sr, (samples * 32767).astype(np.int16))
            subprocess.run(["aplay", f.name], check=True)
    except Exception as e:
        subprocess.run(["espeak", text])

def respond(user_msg):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_msg})
    if len(conversation_history) > 6:
        conversation_history = conversation_history[-6:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    response = llm.create_chat_completion(messages=messages, max_tokens=180, stream=True)
    full_response = ""
    for chunk in response:
        if "content" in chunk["choices"][0]["delta"]:
            token = chunk["choices"][0]["delta"]["content"]
            print(token, end="", flush=True)
            full_response += token
    print()
    conversation_history.append({"role": "assistant", "content": full_response})
    speak(full_response)

def main():
    print("Voice Agent CUDA - Ctrl+C to quit")
    while True:
        try:
            audio = record()
            if audio is None:
                continue
            text = transcribe(audio)
            if not text:
                continue
            print(f"You: {text}")
            respond(text)
        except KeyboardInterrupt:
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
