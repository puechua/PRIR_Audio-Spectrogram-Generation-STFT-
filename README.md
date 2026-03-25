# 🎵 Audio Spectrogram Generation (STFT) — Parallel CPU & GPU

This project implements the Short-Time Fourier Transform (STFT) to generate audio spectrograms, focusing on High-Performance Computing (HPC) and algorithm optimization. The computations are parallelized using **OpenMP** for multi-core CPUs and **CUDA** for NVIDIA GPUs, achieving significant performance improvements over a sequential baseline.

This project was developed by Aleksandra Orzech and Milana Lukasiuk.

## 🛠️ Technologies & Libraries
* **Languages:** C++, Python, CUDA C
* **Parallel Computing:** OpenMP, CUDA
* **FFT Libraries:** FFTW3 (for CPU), cuFFT (for GPU)
* **Scripts & Benchmarking:** Python 

## 🚀 Features & Achievements
* **CPU Parallelization:** Multi-threaded STFT implementation using OpenMP and the FFTW3 library.
* **GPU Acceleration:** Highly parallelized STFT computation leveraging NVIDIA GPUs with CUDA and cuFFT.
* **Performance Optimization:** Achieved up to **4.4× speedup** compared to the sequential baseline through efficient memory management, thread block configuration, and multi-threading.
* **Benchmarking Suite:** Included Python scripts (`benchmark.py`, `prepare_audio.py`) for automated audio preprocessing, execution, and precise performance measurement.

## 📂 Repository Structure
* `stft_openmp.cpp` - CPU parallel implementation using OpenMP.
* `stft_cuda.cu` - GPU parallel implementation using CUDA.
* `prepare_audio.py` - Python script for preprocessing raw audio files into a format ready for C++/CUDA processing.
* `benchmark.py` - Script for running performance benchmarks, executing the algorithms, and comparing execution times.
* `Orzech_Lukasiuk_DS3-2.pdf` - Full project report and detailed performance analysis.
* `*.mp3` - Sample audio files used for testing and benchmarking (e.g., Darude - Sandstorm, Robert Miles - Children).

## 💻 How to Compile and Run

### 1. Audio Preprocessing
Before running the C++/CUDA code, prepare the audio files using Python:
```bash
python prepare_audio.py
