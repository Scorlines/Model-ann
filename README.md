# Prediksi Pendapatan Menggunakan Model ANN

Proyek ini merupakan tugas akhir mata kuliah Intelegensi Buatan yang berfokus pada pembuatan model Artificial Neural Network (ANN) untuk klasifikasi pendapatan berdasarkan dataset Incomes of 30K USA Citizens.

## Deskripsi Tugas

### Langkah Pengerjaan 1: Pembuatan Model ANN – Training & Testing
Tim diwajibkan untuk melakukan eksplorasi dengan memanfaatkan dan memodifikasi model ANN yang telah dipelajari sebelumnya untuk melakukan klasifikasi pada salah satu dari dataset berikut:
- **Incomes of 30K USA Citizens Dataset** (https://www.kaggle.com/datasets/jacopoferretti/incomes-of-30k-usacitizens)

Dataset yang digunakan dalam proyek ini adalah **Incomes of 30K USA Citizens Dataset**, yang merupakan binary-class classification untuk memprediksi apakah pendapatan seseorang >50K atau ≤50K.

### Langkah Pengerjaan 2: Pembuatan Laporan Hasil Eksplorasi
Laporan hasil eksplorasi disusun dalam format PDF dengan struktur bab sebagai berikut:

#### Bab 1. Pendahuluan
- **Subbab 1.1. Pemahaman Dataset** [Bobot CPMK0801: 2.5 poin]
- **Subbab 1.2. Pemrosesan Awal Dataset** [Bobot CPMK0803: 2.5 poin]

#### Bab 2. Landasan Teori
- **Subbab 2.1. Penjelasan Metode dan Model Learning** [Bobot CPMK0801: 2.5 poin]
- **Subbab 2.2. Skema Eksperimen dalam ANN Learning** [Bobot CPMK0801: 2.5 poin]

#### Bab 3. Hasil Eksperimen dan Pembahasan
- **Subbab 3.1. Hasil Pelatihan Model ANN** [Bobot CPMK0803: 7 poin]
- **Subbab 3.2. Hasil Pengujian Model ANN** [Bobot CPMK0803: 7 poin]

#### Bab 4. Kesimpulan [Bobot CPMK0801: 6 poin]

**Format Laporan:**
- Font: Times New Roman, ukuran 12
- Spasi: 1.5
- Margin: Top 2.5cm, Bottom 2.5cm

## Penjelasan Model ANN

### Apa itu Artificial Neural Network (ANN)?
Artificial Neural Network (ANN) adalah model komputasi yang terinspirasi dari struktur dan fungsi jaringan saraf biologis pada otak manusia. ANN terdiri dari lapisan-lapisan neuron yang saling terhubung, di mana setiap koneksi memiliki bobot yang menentukan kekuatan sinyal yang dikirim. ANN digunakan untuk mempelajari pola kompleks dari data dan membuat prediksi berdasarkan pola tersebut.

Dalam konteks klasifikasi, ANN dapat belajar untuk mengenali pola dalam data input dan mengklasifikasikan data baru ke dalam kategori yang sesuai. Model ANN dalam proyek ini digunakan untuk klasifikasi binary: memprediksi apakah pendapatan seseorang lebih dari $50,000 per tahun atau tidak.

### Dataset yang Digunakan
Dataset **Incomes of 30K USA Citizens** diambil dari Kaggle (https://www.kaggle.com/datasets/jacopoferretti/incomes-of-30k-usacitizens). Dataset ini berisi informasi demografis dan ekonomi dari 30,000 warga Amerika Serikat, dengan fitur-fitur seperti:
- **age**: Usia individu
- **workclass**: Jenis pekerjaan (private, self-emp-not-inc, dll.)
- **fnlwgt**: Bobot final sampling
- **education**: Tingkat pendidikan
- **education-num**: Jumlah tahun pendidikan
- **marital-status**: Status pernikahan
- **occupation**: Pekerjaan
- **relationship**: Hubungan dalam keluarga
- **race**: Ras
- **sex**: Jenis kelamin
- **capital-gain**: Keuntungan modal
- **capital-loss**: Kerugian modal
- **hours-per-week**: Jam kerja per minggu
- **native-country**: Negara asal
- **income**: Target variable (≤50K atau >50K)

Dataset ini merupakan binary classification problem dengan 14 fitur input dan 1 target output.

### Preprocessing Data
Sebelum melatih model ANN, data melalui beberapa tahap preprocessing:
1. **Handling Missing Values**: Menghapus baris yang mengandung nilai kosong
2. **Encoding Kategorikal**: Menggunakan Label Encoding untuk mengubah fitur kategorikal menjadi numerik
3. **Feature Scaling**: Menggunakan StandardScaler untuk menormalkan fitur numerikal
4. **Data Splitting**: Membagi data menjadi training (70%), validation (15%), dan testing (15%)

### Arsitektur Model ANN
Model ANN yang dibangun memiliki arsitektur sebagai berikut:

```
Input Layer (14 neurons) → Hidden Layer 1 (128 neurons) → Hidden Layer 2 (64 neurons) → Hidden Layer 3 (32 neurons) → Hidden Layer 4 (16 neurons) → Output Layer (2 neurons)
```

**Detail Lapisan:**
- **Input Layer**: 14 neuron sesuai dengan jumlah fitur
- **Hidden Layer 1**: 128 neuron dengan aktivasi ReLU, Batch Normalization, dan Dropout 0.3
- **Hidden Layer 2**: 64 neuron dengan aktivasi ReLU, Batch Normalization, dan Dropout 0.3
- **Hidden Layer 3**: 32 neuron dengan aktivasi ReLU, Batch Normalization, dan Dropout 0.2
- **Hidden Layer 4**: 16 neuron dengan aktivasi ReLU
- **Output Layer**: 2 neuron dengan aktivasi Softmax untuk klasifikasi binary

**Teknik Regularisasi:**
- **Batch Normalization**: Untuk menstabilkan dan mempercepat training
- **Dropout**: Untuk mencegah overfitting (0.3 pada layer 1-2, 0.2 pada layer 3)

### Hyperparameters dan Training
- **Optimizer**: Adam dengan learning rate awal 0.001
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: Maksimal 100, dengan Early Stopping (patience=15)
- **Learning Rate Scheduling**: ReduceLROnPlateau dengan factor 0.5 dan patience=5

**Callbacks:**
- **Early Stopping**: Menghentikan training jika validation loss tidak membaik selama 15 epoch
- **Reduce Learning Rate**: Mengurangi learning rate jika validation loss tidak membaik selama 5 epoch

### Evaluasi Model
Model dievaluasi menggunakan metrik-metrik berikut:
- **Accuracy**: Persentase prediksi yang benar
- **Precision**: Proporsi prediksi positif yang benar
- **Recall**: Proporsi kasus positif yang terdeteksi
- **F1-Score**: Harmonic mean dari precision dan recall
- **Confusion Matrix**: Matriks yang menunjukkan true positive, false positive, true negative, false negative

Hasil evaluasi ditampilkan untuk data training, validation, dan testing secara terpisah untuk memastikan tidak ada overfitting.

### Visualisasi Hasil
Program menghasilkan tiga visualisasi utama:
1. **Training History**: Grafik akurasi dan loss selama training
2. **Confusion Matrix**: Heatmap matriks konfusi
3. **Performance Comparison**: Perbandingan akurasi dan loss antar dataset

## Struktur Proyek

```
├── income.csv                    # Dataset pendapatan
├── main.py                       # Script utama untuk training dan testing model
├── income_prediction_model.h5    # Model terlatih dalam format HDF5
├── income_prediction_model.keras # Model terlatih dalam format Keras
├── run.bat                       # Batch file untuk menjalankan program
└── README.md                     # Dokumentasi proyek
```

## Cara Menjalankan

### Opsi 1: Menggunakan Batch File (Direkomendasikan)
1. Double-click file `run.bat` untuk menjalankan program secara otomatis.

### Opsi 2: Menggunakan Terminal
1. Pastikan Python 3.13+ terinstall
2. Aktivasi virtual environment (jika ada):
   ```
   .venv\Scripts\activate
   ```
3. Jalankan script:
   ```
   python main.py
   ```

## Output Program

Program akan menghasilkan:
- `training_history.png`: Grafik riwayat pelatihan model
- `confusion_matrix.png`: Matriks konfusi untuk evaluasi model
- `performance_comparison.png`: Perbandingan performa model
- Model terlatih dalam format `.h5` dan `.keras`

## Persyaratan Sistem

- Python 3.13 atau versi lebih baru
- Library yang diperlukan: TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Virtual environment dengan paket terinstall (opsional)

## Catatan

- Dataset akan diunduh otomatis jika file `income.csv` belum tersedia
- Model ANN menggunakan arsitektur yang telah dimodifikasi untuk klasifikasi binary
- Evaluasi model meliputi akurasi, precision, recall, dan F1-score

## Kontribusi

Proyek ini dikembangkan sebagai bagian dari tugas akhir mata kuliah Intelegensi Buatan.
