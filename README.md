# KAN Baseline Multi-Dataset Evaluator

Repositori ini berisi implementasi evaluasi *baseline* menggunakan arsitektur **Kolmogorov-Arnold Networks (KAN)** pada 10 dataset medis publik. Kode ini menggunakan *5-Fold Stratified Cross-Validation* untuk memastikan keandalan hasil pengujian.

## Fitur Utama
- Pelatihan model KAN pada 10 dataset (Breast Cancer, Hepatitis, Liver, dll).
- 5-Fold Stratified CV dengan metrik per-*fold* (Akurasi, Presisi, Recall, F1).
- Ekstraksi 95% Confidence Intervals menggunakan distribusi T.
- Ekspor visualisasi (Box-plot, matriks konfusi) dan laporan JSON/CSV otomatis.

## Cara Menjalankan Secara Lokal

1. Clone repositori ini:
   ```bash
   git clone [https://github.com/USERNAME_GITHUB_MU/NAMA_REPO_MU.git](https://github.com/USERNAME_GITHUB_MU/NAMA_REPO_MU.git)
   cd NAMA_REPO_MU
