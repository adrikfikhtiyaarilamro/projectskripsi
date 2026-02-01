# Endless Runner + Voice Command + Gesture Detection

## Requirements
- Windows 10/11, .NET 9 SDK
- Python 3.10+, pip
- GTK3 runtime (dibutuhkan oleh TCPServer GTK). Jika belum ada, install GTK3 for Windows (MSYS2 `mingw-w64-x86_64-gtk3` atau installer GTK3 runtime).
- (Opsional) GPU + PyTorch CUDA

## Langkah
1) Aktifkan Python env & install deps
```powershell
cd path\to\Endless-Runner-master
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Jalankan TCP Server (port 5005)
```powershell
cd path\to\Endless-Runner-master
dotnet run --configuration Debug
```
- Klik **Start Server** di jendela GTK.
- Pastikan port 5005 listening (`netstat -ano | findstr :5005`).

3) Jalankan Voice Inference GUI
```powershell
cd path\to\Endless-Runner-master\Voice
cd Endless-Runner-master\Voice
..\..\. venv\Scripts\python.exe inference_gui.py
or example
& "C:\Users\Dricky\Downloads\Compressed\Summary v.01\.venv\Scripts\python.exe" inference_gui_transformer.py
```
- Model checkpoint: `checkpoints\transformer\best_model.pth` (MFCC Transformer, 40 koefisien).
- Pastikan konfigurasi host/port GUI mengarah ke `127.0.0.1:5005`.

4) Jalankan Game Unity
- Buka `Game 3D Endless Runner\Aing Kasep.exe`.

5) Alur Kerja
- GUI Voice: rekam/muat audio → model prediksi komando → kirim teks (`left/right/up/down`) via TCP ke server di port 5005.
- TCP Server: terima teks → log di GUI → gerakkan kotak indikator → simulasi keypress ke jendela game → kirim ACK kembali ke GUI.
- Game Unity: menerima input keyboard dan menggerakkan karakter.
- **Cooldown**: Setelah prediksi berhasil, ada cooldown 0.8 detik untuk mencegah deteksi ganda (GUI menampilkan "⏳ Cooldown...").
- **Logging**: Setiap prediksi dicatat di `inference_log_detailed.csv` dengan timestamp, confidence, latency (inference/transport/ACK), dan total response time.

6) Analisis Inferensi
Jalankan notebook analisis komprehensif:
```powershell
cd path\to\Endless-Runner-master\Voice
jupyter notebook analysis_inference.ipynb
```

**Notebook akan menganalisis:**
- 📊 **Latency breakdown**: Inference time vs Transport vs ACK latency
- 🎯 **Accuracy metrics**: Confusion matrix, per-class performance
- 📈 **Decision filter effectiveness**: Commit rate vs Uncertain rate
- 🔧 **Threshold recommendations**: Saran tuning CONF_THRESH, HIGH_CONF_THRESH, TOP2_MARGIN_MIN, cooldown

**Workflow:**
1. Buka GUI, klik "Start Listening", ucapkan 20-30 perintah suara
2. Opsional: buka `inference_log_detailed.csv`, isi kolom "Perintah Suara (Label - Manual)" dengan label benar untuk accuracy metrics
3. Jalankan `analysis_inference.ipynb` untuk lihat hasil
4. Berdasarkan metrik, sesuaikan threshold di `inference_gui.py` (CONF_THRESH, cooldown, dll.)
5. Ulangi test & analisis untuk fine-tuning

## Tips & Troubleshooting
- **Deteksi ganda/duplikat (mis. "right-right" dari satu ucapan)**:
  - Sistem sudah ada **cooldown 0.8 detik** setelah prediksi sukses
  - GUI akan tampilkan "⏳ Cooldown" saat menunggu
  - Jika masih terjadi, tunggu cooldown selesai baru ucapkan perintah berikutnya
  
- **Voice recognition akurasi rendah / aksen kurang bagus**: 
  - **Microphone**: Gunakan headset/desktop mic (lebih baik dari laptop internal mic)
  - **Volume**: Suara harus **jelas dan cukup keras** (RMS > 0.002, cek terminal output)
  - **Pronunciasi**: Ucapkan dengan jelas, tidak perlu aksen sempurna:
    * **"LEF"** atau **"LEFT"** → left
    * **"RAIT"** atau **"RIGHT"** → right  
    * **"AP"** atau **"UP"** → up
    * **"DAUN"** atau **"DOWN"** → down
  - **Environment**: Hindari background noise, ruangan tenang lebih baik
  - **Confidence**: ≥0.4 akan diproses (semakin tinggi semakin yakin model)
  - **Feedback**: Cek terminal untuk `[PROBS]` - lihat distribusi probabilitas semua kelas
  - **Tips**: Jika "Uncertain", coba ulangi dengan lebih keras/jelas, atau lihat top-2 prediksi
- **Venv activate disabled by system**: PowerShell execution policy memblokir script. Jalankan:
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
  atau gunakan Python langsung: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
- **Python path hilang (mis. Python313)**: Jika VS Code/terminal mencari Python di
  `C:\Users\Dricky\AppData\Local\Programs\Python\Python313\python.exe` dan gagal,
  pilih interpreter venv di VS Code: Command Palette → "Python: Select Interpreter" → pilih `.venv\Scripts\python.exe`.
  Alternatif, jalankan GUI via venv langsung: `..\.venv\Scripts\python.exe inference_gui.py`.
- **Port 5005 tidak listening**: pastikan tombol **Start Server** diklik dan netstat menunjukkan LISTENING.
- **GTK error / GUI tidak muncul**: pastikan GTK3 runtime ter-install; jalankan ulang `dotnet run` setelah instalasi.
- **Torch/torchaudio gagal install**: gunakan index PyTorch resmi, contoh CUDA 12.4:
  `pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio`
  atau CPU-only: `pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio`.
- **Game tidak bergerak**: jendela game harus dalam fokus; pastikan TCP log menerima komando.

## 📊 **Data Tracking: Game Status Feedback**

### Alur Data Logging
1. **inference_gui.py** → Merekam audio & run model inference
2. **TCPServer** → Terima command, gerakkan box indikator, kirim keyboard ke game
3. **TCPServer ACK** → Send balik status: `ACK|command|timestamp|game_status`
4. **inference_gui.py** → Log ke CSV termasuk **Game Status** (success/blocked/error)

### CSV Columns
- `Game Status`: Status gerakan di TCPServer:
  - `success`: Gerakan valid (karakter bergerak, atau sudah di edge lane)
  - `blocked`: Gerakan tidak bisa dilakukan (misal "left" tapi sudah di lane paling kiri)
  - `uncertain`: Prediksi uncertain (tidak dikirim ke server)
  - `error`: TCP error

### Contoh Data Real
```csv
ID,Timestamp,Prediksi,Confidence,Game Status
1,2026-01-11 05:40:41,up,0.90,success
2,2026-01-11 05:40:45,Uncertain,0.46,uncertain
3,2026-01-11 05:41:00,left,0.93,blocked
```

### Gunanya
- **Identify false positives**: Prediksi benar tapi game status "blocked" → confidence rendah
- **Verify game integration**: Apakah command sampai ke game?
- **Model improvement**: Data untuk train ulang dengan info game feedback
