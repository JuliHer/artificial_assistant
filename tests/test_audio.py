import json
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_model_input(index=0, rate=16000, frame_ms=32, hop_ms=20, buffer_frames=64):
    # --- Configuración de Rutas ---
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "audio" / "processed" / "dataset"
    json_path = data_dir / "dataset_kora_2025.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_entry = data["files"][index]
    audio_path = data_dir / file_entry["file"]
    
    # --- Procesamiento Idéntico a tu Función ---
    signal, sr = librosa.load(audio_path, sr=rate)
    
    # Conversión de ms a samples
    hop_samples = int(rate * (hop_ms / 1000))
    frame_samples = int(rate * (frame_ms / 1000))
    
    mel_frames = []
    
    # Generar los frames de Mel uno a uno
    for i in range(0, len(signal) - frame_samples, hop_samples):
        frame_audio = signal[i : i + frame_samples]
        
        # Espectrograma de Mel (n_mels=40 según tu código)
        mel = librosa.feature.melspectrogram(
            y=frame_audio, sr=sr, n_fft=512, 
            hop_length=len(frame_audio)+1, # Para obtener 1 solo vector
            n_mels=40
        )
        
        # Logaritmo y Normalización (Z-Score)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = mel[:, 0] # Tomamos la primera columna
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)
        
        mel_frames.append(mel)

    # Convertimos a matriz (Time x Mels)
    full_spec = np.array(mel_frames).T 

    # --- Graficación ---
    plt.figure(figsize=(15, 6))
    
    # Mostramos el espectrograma completo procesado
    # El eje X ahora son "Frames" (cada frame = 20ms)
    plt.imshow(full_spec, aspect='auto', origin='lower', cmap='viridis')
    
    # Dibujar marcas de Wake Word
    # Convertimos los segundos del JSON a número de frame: frame = seg / (hop_ms/1000)
    for interval in file_entry["wake_intervals"]:
        start_frame = interval["start"] / (hop_ms / 1000)
        end_frame = interval["end"] / (hop_ms / 1000)
        
        plt.axvline(x=start_frame, color='red', linestyle='--', linewidth=2)
        plt.axvline(x=end_frame, color='red', linestyle='--', linewidth=2)

    # Dibujar el "Buffer" (Ventana que ve el modelo)
    # Dibujamos un rectángulo que represente el tamaño de buffer_frames (64)
    rect_start = 10 # Un punto arbitrario para ejemplo visual
    plt.gca().add_patch(plt.Rectangle((rect_start, 0), buffer_frames, 39, 
                                     edgecolor='orange', fill=False, linewidth=3, label="Buffer (64 frames)"))

    plt.title(f"Input del Modelo: {file_entry['file']} (Hop: {hop_ms}ms, Window: {frame_ms}ms)")
    plt.xlabel("Frames (Tiempo)")
    plt.ylabel("Mel Filters (Frecuencia)")
    plt.colorbar(label="Z-Score Normalized Energy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    visualize_model_input(index=0)