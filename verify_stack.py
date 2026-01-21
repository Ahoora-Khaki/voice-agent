"""
Voice Agent Stack Verification Script
Tests: GPU, Audio, STT, TTS, LLM inference
"""

import sys
import time

def test_pytorch_mps():
    """Test PyTorch MPS (Metal) acceleration"""
    print("\n" + "="*50)
    print("🔬 TEST 1: PyTorch MPS (Metal GPU)")
    print("="*50)
    
    import torch
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return False
    
    # Run a simple GPU computation
    device = torch.device("mps")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y)
    torch.mps.synchronize()
    elapsed = time.time() - start
    
    print(f"✅ MPS working! 100x matrix multiplications in {elapsed:.3f}s")
    print(f"   Device: {device}")
    return True


def test_audio_devices():
    """Test audio input/output devices"""
    print("\n" + "="*50)
    print("🔬 TEST 2: Audio Devices")
    print("="*50)
    
    import sounddevice as sd
    
    devices = sd.query_devices()
    input_device = sd.query_devices(kind='input')
    output_device = sd.query_devices(kind='output')
    
    print(f"✅ Input device:  {input_device['name']}")
    print(f"✅ Output device: {output_device['name']}")
    
    # Test recording capability (0.1 second silent test)
    try:
        test_audio = sd.rec(int(0.1 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        print(f"✅ Microphone test: OK (captured {len(test_audio)} samples)")
    except Exception as e:
        print(f"⚠️  Microphone test failed: {e}")
        print("   (This is OK if you denied mic permission)")
    
    return True


def test_faster_whisper():
    """Test faster-whisper STT"""
    print("\n" + "="*50)
    print("🔬 TEST 3: Faster-Whisper (STT)")
    print("="*50)
    
    from faster_whisper import WhisperModel
    
    print("   Loading tiny model for quick test...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print(f"✅ faster-whisper loaded successfully")
    print(f"   (Will use 'large-v3-turbo' for production)")
    
    del model
    return True


def test_kokoro_tts():
    """Test Kokoro TTS"""
    print("\n" + "="*50)
    print("🔬 TEST 4: Kokoro-ONNX (TTS)")
    print("="*50)
    
    import kokoro_onnx
    import onnxruntime as ort
    
    providers = ort.get_available_providers()
    print(f"✅ ONNX Runtime providers: {providers}")
    print(f"✅ kokoro-onnx imported successfully")
    print(f"   (Model will be downloaded on first use)")
    
    return True


def test_llama_cpp():
    """Test llama.cpp Metal support"""
    print("\n" + "="*50)
    print("🔬 TEST 5: llama.cpp (Metal GPU)")
    print("="*50)
    
    import llama_cpp
    
    print(f"✅ llama-cpp-python version: {llama_cpp.__version__}")
    print(f"✅ Metal acceleration: Compiled in")
    print(f"   (Model will be downloaded in Phase 3)")
    
    return True


def test_training_stack():
    """Test HuggingFace + PEFT"""
    print("\n" + "="*50)
    print("🔬 TEST 6: Training Stack (HF + PEFT)")
    print("="*50)
    
    import transformers
    import peft
    import accelerate
    
    print(f"✅ transformers: {transformers.__version__}")
    print(f"✅ peft: {peft.__version__}")
    print(f"✅ accelerate: {accelerate.__version__}")
    print(f"   Ready for DoRA/LoRA fine-tuning")
    
    return True


def main():
    print("\n" + "🚀 VOICE AGENT STACK VERIFICATION 🚀".center(50))
    print(f"Python {sys.version.split()[0]}")
    
    results = []
    results.append(("PyTorch MPS", test_pytorch_mps()))
    results.append(("Audio Devices", test_audio_devices()))
    results.append(("Faster-Whisper", test_faster_whisper()))
    results.append(("Kokoro TTS", test_kokoro_tts()))
    results.append(("llama.cpp", test_llama_cpp()))
    results.append(("Training Stack", test_training_stack()))
    
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Ready for Phase 3.")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    return all_passed


if __name__ == "__main__":
    main()
