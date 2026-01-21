"""Text-to-Speech - Kokoro ONNX"""
import tempfile
import subprocess
import platform
from scipy.io import wavfile

class TextToSpeech:
    def __init__(self, model_path="kokoro-v1.0.onnx", voices_path="voices-v1.0.bin", voice="af_heart"):
        self.model_path = model_path
        self.voices_path = voices_path
        self.voice = voice
        self.kokoro = None
    
    def _load(self):
        if self.kokoro is None:
            from kokoro_onnx import Kokoro
            self.kokoro = Kokoro(self.model_path, self.voices_path)
    
    def speak(self, text, speed=1.0):
        """Convert text to speech and play"""
        self._load()
        samples, sr = self.kokoro.create(text, voice=self.voice, speed=speed)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, sr, (samples * 32767).astype('int16'))
            
            if platform.system() == "Darwin":  # Mac
                subprocess.run(["afplay", f.name], check=True)
            else:  # Linux
                subprocess.run(["aplay", f.name], check=True)
    
    def synthesize(self, text, speed=1.0):
        """Return audio samples without playing"""
        self._load()
        return self.kokoro.create(text, voice=self.voice, speed=speed)
