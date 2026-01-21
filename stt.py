"""Speech-to-Text - Supports MLX (Mac) and faster-whisper (NVIDIA)"""
import tempfile
from scipy.io import wavfile

class SpeechToText:
    def __init__(self, model="tiny", backend="auto"):
        self.backend = self._detect_backend(backend)
        self.model = self._load_model(model)
        self.sample_rate = 16000
    
    def _detect_backend(self, backend):
        if backend != "auto":
            return backend
        try:
            import mlx_whisper
            return "mlx"
        except ImportError:
            return "faster-whisper"
    
    def _load_model(self, model):
        if self.backend == "mlx":
            return f"mlx-community/whisper-{model}"
        else:
            from faster_whisper import WhisperModel
            return WhisperModel(model, device="cuda", compute_type="float16")
    
    def transcribe(self, audio):
        """Transcribe audio numpy array to text"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, self.sample_rate, (audio * 32768).astype('int16'))
            
            if self.backend == "mlx":
                import mlx_whisper
                result = mlx_whisper.transcribe(f.name, path_or_hf_repo=self.model)
                return result["text"].strip()
            else:
                segments, _ = self.model.transcribe(f.name)
                return " ".join([s.text for s in segments]).strip()
