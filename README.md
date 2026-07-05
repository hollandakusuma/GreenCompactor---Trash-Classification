# GreenCompactor - Klasifikasi Jenis Sampah dengan CNN (4 Kelas)

## 1. Latar Belakang & Motivasi

**GreenCompactor** adalah alat pemadatan dan penyortiran sampah yang dikembangkan oleh dosen dan mahasiswa Departemen Teknik Elektro, Universitas Maritim Raja Ali Haji (UMRAH). Alat ini dirancang untuk mendukung program kampus yang lebih hijau melalui manajemen sampah yang lebih efisien dan terarah.

Saat ini, GreenCompactor sudah mampu mendeteksi dan memilah material sampah dasar (plastik vs. logam) serta mengompresi sampah secara otomatis. Namun, untuk meningkatkan nilai ekonomis dari sampah yang dikumpulkan dan mengoptimalkan proses daur ulang, diperlukan identifikasi yang lebih detail mengenai **jenis spesifik** dari setiap item sampah yang masuk.

Proyek klasifikasi ini bertujuan untuk mengembangkan **model deep learning berbasis CNN** yang mampu mengidentifikasi 4 kategori sampah yang paling sering ditemukan di lingkungan kampus:

1. **Botol Plastik Bening (PET)** — botol air mineral, minuman bening
2. **Botol Plastik Berwarna** — botol minuman bersoda, jus, kemasan warna-warni
3. **Tutup Botol** — tutup plastik dari berbagai jenis botol
4. **Kaleng Minuman** — kaleng aluminium soda, kopi, energy drink

Dengan model ini, GreenCompactor dapat:
- Mengklasifikasi sampah masuk secara otomatis dengan akurasi tinggi (target =95%)
- Memisahkan sampah ke "bin" yang tepat untuk proses daur ulang lebih lanjut
- Meningkatkan nilai jual sampah berdasarkan jenis dan material (PET, HDPE, aluminium, dll memiliki harga jual yang berbeda)
- Mengumpulkan data statistik penggunaan sampah di kampus untuk program pengelolaan limbah yang lebih baik

---

## **2. Dataset & Metodologi Pengumpulan Data**

### **2.1 Sumber Data**

Dataset untuk proyek ini dikumpulkan dari **Web Scraping (Google Images)** dengan query spesifik per kelas:
- "botol air mineral bening plastik PET"
- "botol minuman berwarna soda Fanta Sprite Pulpy"
- "tutup botol plastik warna-warni"
- "kaleng minuman soda kopi aluminium"


### **2.2 Komposisi Dataset**

**Jumlah Gambar per Kelas:**
- Target awal: 120 gambar asli per kelas (480 total)
- Setelah augmentasi offline: 2.500 gambar per kelas (10.000 total)
- Augmentasi mencakup 20 jenis transformasi (flip, rotasi, resize, brightness, contrast, noise, shearing, grayscale, blur, random crop, dll)

**Penamaan File (Traceable):**
Setiap gambar asli (kode -00) menghasilkan 21 variasi (1 asli + 20 augmentasi):
```
gambar001-00.png  (asli)
gambar001-01.png  (flip horizontal)
gambar001-02.png  (flip vertikal)
...
gambar001-20.png  (random crop)
```


### **2.3 Pembagian Dataset**

Dataset dibagi menjadi tiga subset untuk melatih dan mengevaluasi model:

| Subset | Proporsi | Jumlah Gambar | Penggunaan |
|---|---|---|---|
| **Training** | 70% | 7.000 | Melatih parameter model |
| **Validation** | 15% | 1.500 | Monitoring performa saat training, tuning hyperparameter |
| **Testing** | 15% | 1.500 | Evaluasi final model pada data yang belum pernah dilihat |

Pembagian dilakukan secara **stratified** untuk memastikan distribusi kelas tetap seimbang di ketiga subset.

---

## **3. Arsitektur Model & Eksperimen**

Notebook ini melatih dan membandingkan **3 arsitektur CNN** yang berbeda, semua mengikuti kriteria submission (Sequential, Conv2D, Pooling):

### **Model A — Basic CNN (3 blok Conv+Pool)**
```
Input (160×160×3)
+- Rescaling (1/255)
+- Conv2D(16, 3×3) + ReLU + MaxPooling(2×2)
+- Conv2D(32, 3×3) + ReLU + MaxPooling(2×2)
+- Conv2D(64, 3×3) + ReLU + MaxPooling(2×2)
+- Flatten
+- Dense(128) + ReLU
+- Dense(4) + Softmax
```
**Target:** baseline yang sederhana, cepat training, akurasi ~85%

### **Model B — Deeper CNN (4 blok Conv+Pool + Dropout)**
```
Input (160×160×3)
+- Rescaling (1/255)
+- Conv2D(32, 3×3) + ReLU + MaxPooling(2×2)
+- Conv2D(64, 3×3) + ReLU + MaxPooling(2×2)
+- Conv2D(128, 3×3) + ReLU + MaxPooling(2×2)
+- Conv2D(128, 3×3) + ReLU + MaxPooling(2×2)
+- Flatten
+- Dense(256) + ReLU + Dropout(0.3)
+- Dense(4) + Softmax
```
**Target:** pembelajaran fitur yang lebih dalam, regularisasi dengan Dropout, akurasi ~90%+

### **Model C — CNN dengan BatchNormalization**
```
Input (160×160×3)
+- Rescaling (1/255)
+- Conv2D(32, 3×3) + ReLU + BatchNormalization + MaxPooling(2×2)
+- Conv2D(64, 3×3) + ReLU + BatchNormalization + MaxPooling(2×2)
+- Conv2D(128, 3×3) + ReLU + BatchNormalization + MaxPooling(2×2)
+- Flatten
+- Dense(128) + ReLU + Dropout(0.4)
+- Dense(4) + Softmax
```
**Target:** stabilitas training lebih baik dengan **BatchNormalization**, akurasi optimal ~93-95%

---

## **4. Callback & Training Configuration**

Untuk mencapai akurasi =95% dan menghindari *overfitting*, digunakan 3 callback standar:

| Callback | Fungsi |
|---|---|
| **EarlyStopping** | Menghentikan training jika validation accuracy tidak meningkat selama 5 epoch; restore weights dari epoch terbaik |
| **ReduceLROnPlateau** | Menurunkan learning rate sebesar 50% jika validation loss plateauing (tidak ada improvement 3 epoch), min_lr=1e-6 |
| **ModelCheckpoint** | Menyimpan model weights terbaik berdasarkan validation accuracy; hanya update jika ada peningkatan |

**Hyperparameter:**
- Optimizer: Adam (auto learning rate)
- Loss function: Categorical Crossentropy (multi-class classification)
- Batch size: 32
- Image size: 160×160 pixels
- Max epochs: 25

---

## **5. Preprocessing**

- **Rescaling (1/255):** normalisasi pixel intensity dari range [0, 255] ke [0, 1]

---

## **6. Hasil & Analisis**

Notebook akan menghasilkan:
1. **Perbandingan ketiga model** — tabel akurasi training, validation, dan testing
2. **Plot training history** — 2 grafik per model (akurasi & loss) untuk melihat kurva pembelajaran
3. **Model terbaik terpilih otomatis** berdasarkan test accuracy tertinggi
4. **Inference visualization** — 6 contoh gambar dari test set dengan prediksi model (warna hijau = benar, merah = salah) + confidence score

**Target akurasi:**
- Training =95%
- Validation =94%
- Testing =92-95%

Jika target ini tercapai, model siap untuk diterapkan pada alat GreenCompactor.

---

## **7. Ekspor & Deployment Format**

Model terbaik disimpan dalam **3 format** sesuai kriteria submission:

### **a) SavedModel (TensorFlow native format)**
- Format: `.pb` + `variables/` folder
- Kegunaan: deployment di server, cloud (GCP, AWS), atau sistem embedded yang bisa menjalankan TensorFlow
- Path: `submission/saved_model/`

### **b) TF-Lite (TensorFlow Lite)**
- Format: `.tflite` (binary, ~1-5 MB, sangat ringan)
- Kegunaan: deployment di mobile (Android/iOS), embedded devices (Raspberry Pi, microcontroller), IoT edge
- Plus: label file `label.txt` untuk interpretasi output
- Path: `submission/tflite/`

### **c) TFJS (TensorFlow.js)**
- Format: `model.json` + `group1-shard1of1.bin` (dapat dijalankan di browser/Node.js)
- Kegunaan: web application, dashboard monitoring real-time di browser
- Path: `submission/tfjs_model/`

---
