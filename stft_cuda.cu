#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono>
#include <fstream>

const int SR = 44100;
const int FRAME_SIZE = 1024;
const int HOP_SIZE = 512;
const float PI = 3.14159265359f;

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__global__ void window_kernel(const float* signal, float* frames, int num_frames, int signal_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = num_frames * FRAME_SIZE;

    if (idx < total_points) {
        int frame_idx = idx / FRAME_SIZE;
        int sample_in_frame = idx % FRAME_SIZE;
        int signal_idx = frame_idx * HOP_SIZE + sample_in_frame;
        float val = (signal_idx < signal_len) ? signal[signal_idx] : 0.0f;
        float multiplier = 0.5f * (1.0f - cosf(2.0f * PI * sample_in_frame / (FRAME_SIZE - 1)));
        frames[idx] = val * multiplier;
    }
}

__global__ void magnitude_kernel(cufftComplex* complex_data, float* magnitude, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        magnitude[idx] = sqrtf(complex_data[idx].x * complex_data[idx].x + complex_data[idx].y * complex_data[idx].y);
    }
}

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
        std::cerr << "Wczytano " << num_samples << " próbek (CUDA)." << std::endl;
    }
    return buffer;
}

void save_to_file(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        file.close();
    } else {
        std::cerr << "Nie udalo sie otworzyc pliku do zapisu: " << filename << std::endl;
    }
}

int main() {
    std::vector<float> h_audio = load_signal("input_signal.bin");
    int signal_len = h_audio.size();
    if (signal_len == 0) return 1;

    int num_frames = (signal_len - FRAME_SIZE) / HOP_SIZE + 1;
    int spectrum_size = FRAME_SIZE / 2 + 1;

    float *d_signal, *d_frames, *d_magnitude;
    cufftComplex *d_complex_spectrum;

    CHECK_CUDA(cudaMalloc(&d_signal, signal_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_frames, num_frames * FRAME_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_complex_spectrum, num_frames * spectrum_size * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_magnitude, num_frames * spectrum_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_signal, h_audio.data(), signal_len * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 256;
    int blocks = (num_frames * FRAME_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    window_kernel<<<blocks, threadsPerBlock>>>(d_signal, d_frames, num_frames, signal_len);
    CHECK_CUDA(cudaGetLastError());

    cufftHandle plan;
    if (cufftPlan1d(&plan, FRAME_SIZE, CUFFT_R2C, num_frames) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT Plan creation failed" << std::endl;
        return 1;
    }
    if (cufftExecR2C(plan, d_frames, d_complex_spectrum) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT Exec failed" << std::endl;
        return 1;
    }

    blocks = (num_frames * spectrum_size + threadsPerBlock - 1) / threadsPerBlock;
    magnitude_kernel<<<blocks, threadsPerBlock>>>(d_complex_spectrum, d_magnitude, num_frames * spectrum_size);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << elapsed.count() << std::endl;

    std::vector<float> h_result(num_frames * spectrum_size);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_magnitude, h_result.size() * sizeof(float), cudaMemcpyDeviceToHost));

    save_to_file("output_cuda.bin", h_result);

    cufftDestroy(plan);
    cudaFree(d_signal);
    cudaFree(d_frames);
    cudaFree(d_complex_spectrum);
    cudaFree(d_magnitude);

    return 0;
}
