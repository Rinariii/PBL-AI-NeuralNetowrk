# PBL AI - Implementasi Neural Network
## 👥 Anggota Kelompok
+ Steven Lie Wibowo - G6401231021
+ Daffa Aulia Musyaffa Subyantoro - G6401231028
+ Tristian Yosa - G6401231122
+ Faiz Naufal Huda - G6401231124
+ Daffa Naufal Mumtaz Heryadi - G6401231168

## 🚀 Cara Menjalankan Program
Program ini ditulis menggunakan format [Jupyter Notebook](https://jupyter.org/) sehingga paling baik dijalankan di lingkungan seperti [Google Colab](https://colab.research.google.com/) atau instalasi Jupyter lokal.

1. Membuka Notebook
+ **Google Colab**: 
    + Pilih `File` -> `Open Notebook`, atau tekan `CTRL+O`
    + Pilih file `.ipnyb` yang berisi program ini. Program akan otomatis dimuat pada Google Colab. 
+ **Jupyter secara Lokal**: 
    + Jalankan server Jupyter Notebook anda
    + Buka file `.ipynb` yang berisi program ini. 
    + Install library yang dibutuhkan dengan menjalankan perintah ini pada terminal anda: 
    ```bash
    pip install -r requirements.txt
    ```
    + Jika anda **tidak memiliki** file `requirements.txt`, jalankan perintah ini pada terminal anda untuk meng-install library yang dibutuhkan: 
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn
    ```

2. Menjalankan Cell
+ Jalankan cell secara berurutan, dari atas ke bawah. Ini akan menjalankan program menggunakan pengaturan 


## ⚙️ Detail Program

### Tahapan Utama dalam Program:

1.  **Import Library:** Mengimpor semua *library* yang dibutuhkan.
2.  **Membuat Fungsi Neural Network:** Mendefinisikan kelas `SimpleNeuralNetwork` yang mengimplementasikan arsitektur neural network 2 layer dengan fungsi aktivasi **ReLU** dan heuristik **He initialization**.
3.  **Load Dataset:** Memuat *dataset* yang digunakan untuk pelatihan dan pengujian.
4.  **Exploratory Data Analysis (EDA):** Menampilkan matriks korelasi dan *boxplot* fitur.
5.  **Preprocessing Data:** Memisahkan fitur (`X`) dan target (`y` - **`SoilMoisture`**), membagi data menjadi *training/testing* set, dan melakukan **Standard Scaling** pada data.
6.  **Membuat Model dan Training:** Menginisialisasi model dengan ukuran *input* 4 dan 8 lapisan tersembunyi, lalu melatihnya dengan `epochs=5000` dan *learning rate* `lr=0.0001`.
7.  **Testing dan Evaluasi:** Membuat prediksi, mengembalikan hasil ke skala asli, dan mengevaluasi performa menggunakan metrik **MSE, MAE, dan R2 Score**.

---

## 💻 Konfigurasi Model (Class: `SimpleNeuralNetwork`)


| Parameter | Nilai/Deskripsi |
| :--- | :--- |
| **`input_size`** | 4 (Fitur: `Temperature`, `Humidity`, `Rainfall`, `CloudCover`) |
| **`hidden_size`** | 8 |
| **`output_size`** | 1 (Target: `SoilMoisture`) |
| **Fungsi Aktivasi** | ReLU |
| **Inisialisasi Bobot** | He Initialization |
| **Epochs (Pelatihan)** | 5000 |
| **Learning Rate (`lr`)** | 0.0001 |

---

## 📊 Dataset

| Kolom | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `Temperature` | float64 | Suhu (°C) |
| `Humidity` | float64 | Kelembaban (skala 0 - 100%) |
| `Rainfall` | float64 | Curah Hujan (mm) |
| `CloudCover` | float64 | Tutupan Awan (skala 0 - 100%) |
| `SoilMoisture` | float64 | Kelembaban Tanah (Target) (skala 0 - 100%) |
