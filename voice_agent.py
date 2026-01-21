import numpy as np
import sounddevice as sd
import webrtcvad
import tempfile
import subprocess
import collections
import time
from scipy.io import wavfile
import mlx_whisper
from mlx_lm import load, stream_generate

# === CONFIG ===
SAMPLE_RATE = 16000  # WebRTC VAD requires 8000, 16000, 32000, or 48000
FRAME_DURATION_MS = 30  # WebRTC VAD works with 10, 20, or 30ms frames
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples per frame
SILENCE_FRAMES = 10  # 10 frames × 30ms = 300ms silence to stop

# WebRTC VAD aggressiveness (0-3, higher = more aggressive filtering)
VAD_MODE = 3  # Most aggressive - filters out more non-speech

# Models
WHISPER_MODEL = "mlx-community/whisper-tiny"
LLM_MODEL = "mlx-community/gemma-2-2b-it-4bit"

SYSTEM_PROMPT = """You are a brutally witty voice AI with zero filter but genuine care. Every response flows naturally in one seamless paragraph.

Start with a savage roast or joke about their question, then immediately shift to warm helpful problem-solving with practical advice, and end by offering one specific action you can take for them.

Keep it under 5 sentences total. No bullet points, no numbering, no formatting. Just natural flowing speech like a brilliant friend who roasts you then helps you. Never say "As an AI" or break character."""

# === INIT ===
print("🔄 Loading models...")
vad = webrtcvad.Vad(VAD_MODE)
model, tokenizer = load(LLM_MODEL)
conversation_history = []
print("✅ Ready!")

def record():
    """Record with WebRTC VAD - stops after 300ms silence"""
    print("🎤 Listening...")
    
    audio_buffer = []
    ring_buffer = collections.deque(maxlen=SILENCE_FRAMES)
    speech_started = False
    
    def callback(indata, frames, time_info, status):
        audio_buffer.append(indata.copy())
    
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',  # WebRTC VAD needs int16
        callback=callback,
        blocksize=FRAME_SIZE
    )
    
    stream.start()
    start_time = time.time()
    
    try:
        while time.time() - start_time < 15:  # Max 15 seconds
            time.sleep(FRAME_DURATION_MS / 1000)  # Wait for one frame
            
            if not audio_buffer:
                continue
            
            # Get latest frame
            frame = audio_buffer[-1].flatten().tobytes()
            
            # WebRTC VAD check
            try:
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
            except:
                continue
            
            if is_speech:
                speech_started = True
                ring_buffer.clear()
            elif speech_started:
                ring_buffer.append(1)  # Count silence frames
                if len(ring_buffer) >= SILENCE_FRAMES:
                    print("⏹️ Silence detected")
                    break
                    
    finally:
        stream.stop()
        stream.close()
    
    if not audio_buffer or not speech_started:
        return None
    
    # Combine all audio
    audio = np.concatenate(audio_buffer).flatten()
    print(f"✅ Recorded {len(audio)/SAMPLE_RATE:.1f}s")
    
    # Convert to float32 for Whisper
    return audio.astype(np.float32) / 32768.0

def transcribe(audio):
    """Transcribe audio with Whisper"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, SAMPLE_RATE, (audio * 32768).astype(np.int16))
        result = mlx_whisper.transcribe(f.name, path_or_hf_repo=WHISPER_MODEL)
    return result["text"].strip()

def speak(text):
    """Speak text with Kokoro TTS"""
    try:
        from kokoro_onnx import Kokoro
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        samples, sr = kokoro.create(text, voice="af_heart", speed=1.0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, sr, (samples * 32767).astype(np.int16))
            subprocess.run(["afplay", f.name], check=True)
    except Exception as e:
        print(f"TTS error: {e}, using macOS say")
        subprocess.run(["say", text])

def respond(user_msg):
    """Generate and speak response"""
    global conversation_history
    
    if not conversation_history:
        full_msg = f"{SYSTEM_PROMPT}\n\nUser: {user_msg}"
    else:
        full_msg = user_msg
    
    conversation_history.append({"role": "user", "content": full_msg})
    if len(conversation_history) > 6:
        conversation_history = conversation_history[-6:]
    
    prompt = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("💬 ", end="", flush=True)
    
    full_response = ""
    for r in stream_generate(model, tokenizer, prompt, max_tokens=180):
        token = r.text
        print(token, end="", flush=True)
        full_response += token
    
    print()
    conversation_history.append({"role": "assistant", "content": full_response})
    
    # Speak entire response
    print("🔊 Speaking...")
    speak(full_response)
    print("✅ Done")

def main():
    print("\n🎙️ Voice Agent with WebRTC VAD")
    print("=" * 40)
    print("Speak naturally - stops after 0.3s silence")
    print("Ctrl+C to quit\n")
    
    while True:
        try:
            audio = record()
            if audio is None:
                print("❌ No speech detected")
                continue
            
            print("📝 Transcribing...")
            text = transcribe(audio)
            if not text:
                continue
            
            print(f"You: {text}")
            respond(text)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
