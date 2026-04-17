from src.audio.engine.audio_engine import AudioEngine
from src.audio.engine.states import AudioState
from src.audio.io.microphone import Microphone
from src.audio.processing.energy_detector import EnergyDetector
from src.audio.processing.recorder import Recorder
from src.audio.processing.soft_vad import SoftVAD
from src.audio.processing.silero_vad import SileroVAD
from src.audio.processing.wake_word import WakeWordController
from pathlib import Path
import tf_keras as keras
import tensorflow as tf
import os

hops = 20
frames = 32
mic = Microphone()
energy = EnergyDetector(change_frames=32)
sileroVAD = SileroVAD("models/silero_vad/silero_vad.jit")
vad = SoftVAD(sileroVAD, min_speech_frames=3, hangover_frames=64)

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

wakeword_model_path = str(project_root / "models" / "wakeword_detector" / f"KORA_w3.keras")
if os.path.exists(wakeword_model_path):
    wakeword_model = keras.models.load_model(wakeword_model_path)
wake_word = WakeWordController(wakeword_model, threshold=0.8)
recorder = Recorder(energy, silence_ms=500, hop_ms=hops)

audioEngine = AudioEngine(mic, energy, vad, wake_word, recorder, hop_ms=hops, frame_ms=frames)

audioEngine.start()

