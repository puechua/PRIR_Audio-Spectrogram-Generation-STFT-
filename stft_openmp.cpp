// stft_openmp.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <omp.h>
#include <chrono>
#include <fstream> 

const int SR = 44100;
const int FRAME_SIZE = 1024;
const int HOP_SIZE = 512;
const float PI = 3.14159265359f;

std::vector<float> load_signal(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "BLAD: Nie znaleziono pliku " << filename << "!" << std::endl;
        std::cerr << "Uruchom najpierw skrypt: python3 prepare_audio.py" << std::endl;
        exit(1);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    int num_samples = size / sizeof(float);
    std::vector<float> buffer(num_samples);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Wczytano " << num_samples << " probek (OpenMP)." << std::endl;
    }
    return buffer;
}

void apply_hann_window(double* frame, int size) {
    for (int i = 0; i < size; ++i) {
        double multiplier = 0.5 * (1 - cos(2 * PI * i / (size - 1)));
        frame[i] *= multiplier;
    }
}

void save_to_file(const std::string& filename, const std::vector<double>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
        file.close();
    } else {
        std::cerr << "Nie udalo sie otworzyc pliku do zapisu!" << std::endl;
    }
}

int main() {
    std::vector<float> raw_audio = load_signal("input_signal.bin");
    int signal_len = raw_audio.size();
    if (signal_len == 0) return 1;

    std::vector<double> audio(signal_len);
    for(int i=0;i<signal_len;++i) audio[i] = static_cast<double>(raw_audio[i]);

    int num_frames = (signal_len - FRAME_SIZE) / HOP_SIZE + 1;
    int spectrum_size = FRAME_SIZE / 2 + 1;
    std::vector<double> spectrogram(num_frames * spectrum_size);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        double* in = (double*) fftw_malloc(sizeof(double) * FRAME_SIZE);
        fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * spectrum_size);
        fftw_plan p;
        #pragma omp critical
        {
            p = fftw_plan_dft_r2c_1d(FRAME_SIZE, in, out, FFTW_ESTIMATE);
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < num_frames; ++i) {
            int start_idx = i * HOP_SIZE;
            for (int j = 0; j < FRAME_SIZE; ++j) {
                if (start_idx + j < signal_len)
                    in[j] = audio[start_idx + j];
                else
                    in[j] = 0.0;
            }
            apply_hann_window(in, FRAME_SIZE);
            fftw_execute(p);
            for (int j = 0; j < spectrum_size; ++j) {
                double mag = sqrt(out[j][0]*out[j][0] + out[j][1]*out[j][1]);
                spectrogram[i * spectrum_size + j] = mag;
            }
        }

        #pragma omp critical
        {
            fftw_destroy_plan(p);
        }
        fftw_free(in);
        fftw_free(out);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << elapsed.count() << std::endl;

    save_to_file("output_omp.bin", spectrogram);
    return 0;
}
