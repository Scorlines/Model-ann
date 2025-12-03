# Prediksi Pendapatan Menggunakan Model ANN

Proyek ini merupakan tugas akhir mata kuliah Intelegensi Buatan yang berfokus pada pembuatan model Artificial Neural Network (ANN) untuk klasifikasi pendapatan berdasarkan dataset Incomes of 30K USA Citizens.

## Deskripsi Tugas

### Langkah Pengerjaan 1: Pembuatan Model ANN – Training & Testing
Tim diwajibkan untuk melakukan eksplorasi dengan memanfaatkan dan memodifikasi model ANN yang telah dipelajari sebelumnya untuk melakukan klasifikasi pada salah satu dari dataset berikut:
- **Incomes of 30K USA Citizens Dataset** (https://www.kaggle.com/datasets/jacopoferretti/incomes-of-30k-usacitizens)
- **Iris Flower Dataset** (https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
- **Disease Symptoms and Patient Profile Dataset** (https://www.kaggle.com/datasets/uom190346a/diseasesymptoms-and-patient-profile-dataset)

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