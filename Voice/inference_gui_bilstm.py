import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import math
import socket
import os
import csv
import time
from collections import deque

# === TCP Settings ===
TCP_HOST = '127.0.0.1'
TCP_PORT = 5005

socketClient = None

# === Model Settings ===
SAMPLE_RATE = 16000
DURATION = 1.0 
N_MFCC = 40  
N_MELS = 40
NUM_CLASSES = 4
LABELS = ['down', 'left', 'right', 'up']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BiLSTM Pink 40 - pure LSTM model
MODEL_PATH = os.path.join(BASE_DIR, r'../../Endless-Runner-master/checkpoints/bilstm/bestmodel.pth')

# === Configuration Pure BiLSTM ===
CONF_THRESH = 0.30
HIGH_CONF_THRESH = 0.45
TOP2_MARGIN_MIN = 0.03
AGREE_WINDOW = 3
AGREE_MIN = 1

# MFCC parameters for 16kHz
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160

# === BiLSTM Settings ===
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2

# === Model Definition ===

# BiLSTM 
class MFCC_LSTM(nn.Module):
    def __init__(
        self,
        n_mfcc: int = 40,
        hidden: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,n_mfcc] or [B,n_mfcc,T]
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {tuple(x.shape)}")
        if x.shape[-1] != self.lstm.input_size and x.shape[1] == self.lstm.input_size:
            x = x.transpose(1, 2).contiguous()
        
        lstm_out, _ = self.lstm(x)
        feat = lstm_out[:, -1, :]  # Take last timestep
        return self.head(feat)

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MFCC_LSTM(
    n_mfcc=N_MFCC, 
    num_classes=NUM_CLASSES,
    hidden=LSTM_HIDDEN,
    num_layers=LSTM_LAYERS,
    dropout=LSTM_DROPOUT
)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✓ Pure BiLSTM model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    print(f"  Please check if the model file exists at: {MODEL_PATH}")

model.to(device)
model.eval()

# === MFCC Transform ===
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "center": True,
        "f_min": 0.0,
        "f_max": SAMPLE_RATE / 2
    }
).to(device)

def send_tcp_command(command):
    """Send command via TCP and wait for ACK."""
    try:
        if socketClient:
            full_msg = f"{command.upper()}\n"
            t0_send = time.perf_counter()
            socketClient.sendall(full_msg.encode('utf-8'))
            t1_send = time.perf_counter()
            send_duration = t1_send - t0_send

            ack_latency = None
            game_status = "unknown"
            try:
                socketClient.settimeout(1.0)
                buf = b""
                while True:
                    chunk = socketClient.recv(1024)
                    if not chunk:
                        break
                    buf += chunk
                    if b"\n" in buf or buf.startswith(b"ACK"):
                        break
                if buf:
                    line = buf.split(b"\n")[0].decode("utf-8", errors="ignore").strip()
                    if line.startswith("ACK"):
                        t_ack = time.perf_counter()
                        ack_latency = t_ack - t1_send
                        parts = line.split("|")
                        if len(parts) >= 4:
                            game_status = parts[3]
                        print(f"[TCP] Received: {line} (ACK latency {ack_latency*1000:.2f} ms, Status: {game_status})")
            except Exception as e:
                print(f"[TCP] ACK wait error: {e}")
            finally:
                try:
                    socketClient.settimeout(None)
                except Exception:
                    pass

            print(f"[TCP] Sent: {command}")
            return send_duration, ack_latency, game_status
    except Exception as e:
        print(f"[TCP] Error: {e}")
    return 0.0, None, "error"

import soundfile as sf 
SAVE_DIR = os.path.join(BASE_DIR, "captured_samples")
os.makedirs(SAVE_DIR, exist_ok=True)
save_samples_var = None

def save_audio_sample(audio_np, label, prob):
    """Save the audio sample to disk"""
    if not (save_samples_var and save_samples_var.get()):
        return
        
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(SAVE_DIR, f"{timestamp}_{label}_{prob:.2f}.wav")
        data_to_save = audio_np.flatten()
        sf.write(filename, data_to_save, SAMPLE_RATE)
        print(f"[DATA] Saved {filename}")
    except Exception as e:
        print(f"[DATA] Failed to save sample: {e}")

# === Logging Setup ===
LOG_FILE = "inference_log_bilstm.csv"
experiment_id = 0

def init_csv_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = [
                "ID Percobaan", "Timestamp", "Perintah Suara", "Prediksi Model", 
                "Status", "Confidence", "Inference Time (ms)", "Transport Latency (ms)", 
                "Server ACK Latency (ms)", "Total Response Time (ms)", "Game Status"
            ]
            writer.writerow(headers)
            print(f"[LOG] Created {LOG_FILE}")

init_csv_log()

def log_inference(prediction, confidence, inf_time, transport_time, ack_latency=None, game_status="unknown"):
    global experiment_id
    experiment_id += 1
    total_time = inf_time + transport_time + (ack_latency if ack_latency is not None else 0.0)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                experiment_id, timestamp, "", prediction, "", f"{confidence:.2f}", 
                f"{inf_time*1000:.2f}", f"{transport_time*1000:.2f}", 
                f"{(ack_latency*1000 if ack_latency is not None else 0.0):.2f}",
                f"{total_time*1000:.2f}", game_status
            ])
    except Exception as e:
        print(f"[LOG] Error writing to CSV: {e}")

is_listening = False
last_prediction_time = 0
COOLDOWN_SECONDS = 0.8
pred_history = deque(maxlen=AGREE_WINDOW)

def continuous_listen():
    """Continuously listen and predict"""
    global is_listening, last_prediction_time
    window_size = int(0.05 * SAMPLE_RATE)

    while is_listening:
        time_since_last = time.time() - last_prediction_time
        if time_since_last < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - time_since_last
            result_var.set(f"⏳ Cooldown {remaining:.1f}s...")
            time.sleep(0.1)
            continue
        
        frames = []

        def callback(indata, frames_count, time, status):
            amplified = indata
            rms = np.sqrt(np.mean(amplified**2))
            volume_level.set(min(rms * 500, 100))
            frames.append(amplified.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=window_size, callback=callback):
            sd.sleep(int(DURATION * 1000))

        audio = np.concatenate(frames, axis=0).T
        audio = np.clip(audio, -1.0, 1.0)
        
        if audio.shape[1] < SAMPLE_RATE:
            pad_length = SAMPLE_RATE - audio.shape[1]
            audio = np.pad(audio, ((0, 0), (0, pad_length)), mode='constant')
        elif audio.shape[1] > SAMPLE_RATE:
            audio = audio[:, :SAMPLE_RATE]
        
        global_rms = np.sqrt(np.mean(audio**2))

        if global_rms < 0.002: 
            result_var.set(f"No sound detected (RMS: {global_rms:.4f})")
            continue

        waveform = torch.tensor(audio, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        max_val = waveform.abs().max()
        if max_val > 1e-6:
            waveform = waveform / max_val
        
        features = mfcc_transform(waveform) 
        features = features.transpose(1, 2).contiguous()  
        features = features.to(device)

        t0_inf = time.perf_counter()
        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            max_prob, pred = torch.max(probs, dim=1)

            prob_str = " | ".join([f"{LABELS[i]}: {probs[0][i].item():.3f}" for i in range(len(LABELS))])
            print(f"[PROBS] {prob_str}")
            print(f"[PRED] {LABELS[pred.item()].upper()} with confidence {max_prob.item():.4f}")

            t1_inf = time.perf_counter()
            inference_dur = t1_inf - t0_inf
            transport_dur = 0.0

            probs_sorted, indices = torch.sort(probs[0], descending=True)
            top1_label = LABELS[indices[0].item()]
            top2_label = LABELS[indices[1].item()]
            top1_prob = probs_sorted[0].item()
            top2_prob = probs_sorted[1].item()
            margin = top1_prob - top2_prob

            should_commit = False
            agree_count = 1
            if top1_prob >= HIGH_CONF_THRESH:
                should_commit = True
            else:
                pred_history.append((top1_label, top1_prob))
                agree_count = sum(1 for lbl, _p in pred_history if lbl == top1_label)
                if margin >= TOP2_MARGIN_MIN and top1_prob >= CONF_THRESH and agree_count >= AGREE_MIN:
                    should_commit = True

            if not should_commit:
                result_var.set(f"❓ Uncertain: {top1_label}({top1_prob:.2f}) vs {top2_label}({top2_prob:.2f})")
            else:
                prediction = top1_label
                result_var.set(f"✅ {prediction.upper()} ({top1_prob:.2f})")
                
                send_dur, ack_latency, game_status = send_tcp_command(prediction)
                transport_dur = send_dur
                
                last_prediction_time = time.time()
                print(f"[COOLDOWN] Started {COOLDOWN_SECONDS}s cooldown after '{prediction}' (prob={top1_prob:.2f}, margin={margin:.2f}, agree={agree_count})")
                
            current_pred = (prediction if should_commit else "Uncertain")
            try:
                game_status_log = game_status if should_commit else "uncertain"
                log_inference(current_pred, top1_prob, inference_dur, transport_dur, ack_latency if should_commit else None, game_status_log)
            except NameError:
                log_inference(current_pred, top1_prob, inference_dur, transport_dur, None)

            save_audio_sample(audio, LABELS[pred.item()], max_prob.item())


def start_listening():
    """Start listening mode"""
    global is_listening, socketClient

    if not is_listening:
        try:
            socketClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socketClient.connect((TCP_HOST, TCP_PORT))
            print("[TCP] Connected")
        except Exception as e:
            print(f"[TCP] Connection failed: {e}")
            print("[TCP] Running without TCP server (voice recognition still works)")
            socketClient = None

        is_listening = True
        result_var.set("🎧 Listening...")
        threading.Thread(target=continuous_listen, daemon=True).start()


def stop_listening():
    """Stop listening mode"""
    global is_listening, socketClient, last_prediction_time
    is_listening = False
    last_prediction_time = 0

    if socketClient:
        try:
            socketClient.close()
            print("[TCP] Closed")
        except Exception as e:
            pass

    socketClient = None
    result_var.set("⏹ Stopped Listening.")


# GUI
root = tk.Tk()
root.title("Voice Command - Pure BiLSTM")
root.configure(bg="#2e2e2e")

volume_level = tk.DoubleVar()
result_var = tk.StringVar()
result_var.set("Press Start to begin")

style = ttk.Style()
style.theme_use("clam")
style.configure("TFrame", background="#2e2e2e")
style.configure("TLabel", background="#2e2e2e", foreground="white", font=("Arial", 12))
style.configure("TButton", background="#444", foreground="white", font=("Arial", 10))
style.map("TButton", background=[("active", "#666")])
style.configure("TProgressbar", background="#4caf50")

frame = ttk.Frame(root, padding=20)
frame.grid()

title = ttk.Label(frame, text=f"Voice Command - Pure BiLSTM (Pink 40)", font=("Arial", 16, "bold"))
title.grid(column=0, row=0, columnspan=2, pady=10)

model_info = ttk.Label(frame, text=f"Model: Pure BiLSTM (MFCC_LSTM) | SR: {SAMPLE_RATE}Hz | MFCC: {N_MFCC}", 
                       font=("Arial", 9))
model_info.grid(column=0, row=1, columnspan=2, pady=5)

start_btn = ttk.Button(frame, text="▶ Start Listening", command=start_listening)
start_btn.grid(column=0, row=2, pady=10, padx=5)

stop_btn = ttk.Button(frame, text="⏹ Stop Listening", command=stop_listening)
stop_btn.grid(column=1, row=2, pady=10, padx=5)

volume_label = ttk.Label(frame, text="Volume Level:", font=("Arial", 10))
volume_label.grid(column=0, row=3, columnspan=2, pady=(10, 0))

progress = ttk.Progressbar(frame, orient='horizontal', length=300,
                           mode='determinate', maximum=100,
                           variable=volume_level)
progress.grid(column=0, row=4, columnspan=2, pady=5)

result_label = ttk.Label(frame, textvariable=result_var, font=("Arial", 14, "bold"))
result_label.grid(column=0, row=5, columnspan=2, pady=15)

classes_label = ttk.Label(frame, text=f"Classes: {', '.join(LABELS)}", font=("Arial", 9))
classes_label.grid(column=0, row=6, columnspan=2, pady=5)

save_samples_var = tk.BooleanVar(value=False)
save_check = ttk.Checkbutton(frame, text="Save Recorded Samples", 
                             variable=save_samples_var, onvalue=True, offvalue=False)
save_check.grid(column=0, row=7, columnspan=2, pady=10)

def update_gui():
    root.after(50, update_gui)

update_gui()

print(f"\n{'='*60}")
print(f"Voice Command Classifier - PURE BiLSTM")
print(f"{'='*60}")
print(f"Model: Pure BiLSTM (MFCC_LSTM - Pink 40)")
print(f"Conf Threshold: {CONF_THRESH}")
print(f"High Conf Threshold: {HIGH_CONF_THRESH}")
print(f"Model Path: {MODEL_PATH}")
print(f"Device: {device}")
print(f"Sample Rate: {SAMPLE_RATE} Hz")
print(f"MFCC: {N_MFCC} coefficients")
print(f"Classes: {LABELS}")
print(f"{'='*60}\n")

root.mainloop()
