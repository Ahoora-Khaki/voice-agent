"""Voice Activity Detection - WebRTC VAD"""
import webrtcvad
import collections
import numpy as np

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, frame_ms=30, silence_ms=300, aggressiveness=3):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.silence_frames = int(silence_ms / frame_ms)
        self.vad = webrtcvad.Vad(aggressiveness)
    
    def is_speech(self, audio_frame):
        """Check if audio frame contains speech"""
        return self.vad.is_speech(audio_frame, self.sample_rate)
    
    def get_frame_size(self):
        return self.frame_size
    
    def get_silence_frames(self):
        return self.silence_frames
