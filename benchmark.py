import matplotlib
matplotlib.use('Agg')

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import csv
from pathlib import Path

#parametry STFT
SR = 44100 #częstotliwość próbkowania (sample rate)
FRAME_SIZE = 1024 #4ozmiar ramki czasowej
HOP_SIZE = 512 #przesunięcie ramki (overlap 50%)
SPECTRUM_SIZE = FRAME_SIZE // 2 + 1  #rozmiar widma (dla rfft)

#ścieżki do plików audio
SONGS = [
    "The Four Seasons - Big Girls Don't Cry (Official Audio).mp3",
    "Darude - Sandstorm.mp3",
    "Robert Miles - Children [Dream Version].mp3"
]

#lista wątków OpenMP
OMP_THREADS = [1, 2, 4, 8, 12]

#ścieżki do skompilowanych programów C++
BIN_OMP = "./stft_omp"
BIN_CUDA = "./stft_cuda"

#katalog wyników
OUT_DIR = Path("benchmark_results")
OUT_DIR.mkdir(exist_ok=True)
CSV_FILE = OUT_DIR / "results.csv"

#timeout uruchomienia programów (sekundy)
RUN_TIMEOUT = 600



#konwertowanie pliku audio (mp3/wav) do surowego formatu binarnego (float32) przy użyciu pomocniczego skryptu pythonowego
def prepare_input_via_script(audio_path, output_bin="input_signal.bin"):
    cmd = f"python3 prepare_audio.py --input \"{audio_path}\" --output \"{output_bin}\""
    print(f"Przygotowanie input: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("Błąd przygotowania audio:", res.stderr)
        return False
    return True





#uruchomienie zew programu C++ - przechwycenie wyjścia i parsowanie czasu wykonania
def run_cpp_program(command, env=None):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"Timeout: {command} przekroczył {RUN_TIMEOUT}s")
        return 0.0

    if result.returncode != 0:
        print(f"Błąd uruchamiania: {command}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        return 0.0

    out_lines = result.stdout.strip().splitlines()
    if not out_lines:
        return 0.0

    last = out_lines[-1].strip()
    try:
        return float(last)
    except ValueError:
        for line in reversed(out_lines):
            try:
                return float(line.strip())
            except:
                continue
        print(f"Nie znaleziono czasu w wyjściu polecenia: {command}\nOstatnie linie:\n{out_lines[-5:]}")
        return 0.0


#referencyjna implementacja STFT w Pythonie (NumPy) - mierzenie czasu samej transformacji
def compute_python_stft_time(binfile="input_signal.bin"):
    audio = np.fromfile(binfile, dtype=np.float32)
    SIGNAL_LEN = len(audio)
    if SIGNAL_LEN < FRAME_SIZE:
        print("Uwaga: sygnał krótszy niż FRAME_SIZE")
    start = time.time()
    frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME_SIZE)[::HOP_SIZE]
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    spec_py = np.abs(np.fft.rfft(frames * window, axis=1))
    elapsed = time.time() - start
    return elapsed, spec_py



#wczytanie wynikowego spektrogramu z pliku binarnego + formowanie go w macierz; przycinanie danych w przypadku nadmiarowych bajtów
def load_spectrogram_from_file(path, dtype, num_frames):
    if not Path(path).exists():
        return None
    data = np.fromfile(path, dtype=dtype)
    expected = num_frames * SPECTRUM_SIZE
    if data.size != expected:
        print(f"Uwaga: rozmiar {path} = {data.size}, oczekiwano {expected}. Przycinam/przekształcam.")
        data = data[:expected]
    mat = data.reshape((num_frames, SPECTRUM_SIZE)).T
    return mat



#generowanie wykresu słupkowego - porównanie czasu wykonania
def plot_performance(song_label, duration_s, py_time, omp_times, cuda_time, threads, outdir):
    labels = ['Python'] + [f'OMP({t})' for t in threads] + ['CUDA']
    times = [py_time] + omp_times + [cuda_time]
    plt.figure(figsize=(10,6))
    colors = ['red'] + ['orange']*len(threads) + ['green']
    bars = plt.bar(labels, times, color=colors)
    plt.ylabel('Czas wykonania [s]')
    plt.title(f'Porównanie wydajności STFT: {song_label} ({duration_s:.1f}s)')
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}s', ha='center', va='bottom', fontsize=8)
    out_path = Path(outdir) / f"performance_{song_label}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Zapisano: {out_path}")



#wizualne porównanie spektrogramów
def plot_validation_spectrograms(song_label, spec_py, spec_omp, spec_cuda, outdir):
    fig, ax = plt.subplots(3, 1, figsize=(10,12))
    def plot_spec(ax, data, title):
        if data is None:
            ax.text(0.5,0.5,'BRAK DANYCH', ha='center', va='center')
            ax.set_title(title)
            return
        img = ax.imshow(20*np.log10(np.abs(data)+1e-6), aspect='auto', origin='lower', cmap='inferno')
        ax.set_title(title)
        return img
    plot_spec(ax[0], spec_py.T if spec_py is not None else None, 'Python (reference)')
    plot_spec(ax[1], spec_omp, 'C++ OpenMP')
    plot_spec(ax[2], spec_cuda, 'C++ CUDA')
    plt.tight_layout()
    out_path = Path(outdir) / f"spectrograms_{song_label}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Zapisano: {out_path}")


#główna logika testu dla jednego utworu: 1.Python, 2.OpenMP, 3.CUDA
def benchmark_one_song(song_path, song_label, outdir):
    os.makedirs(outdir, exist_ok=True)
    ok = prepare_input_via_script(song_path, output_bin="input_signal.bin")
    if not ok:
        print(f"Pominięto {song_path} z powodu błędu przygotowania.")
        return None

    audio = np.fromfile("input_signal.bin", dtype=np.float32)
    SIGNAL_LEN = len(audio)
    NUM_FRAMES = (SIGNAL_LEN - FRAME_SIZE) // HOP_SIZE + 1
    duration_s = SIGNAL_LEN / SR
    print(f"Sygnał: {SIGNAL_LEN} próbek ({duration_s:.2f}s), oczekiwane ramki: {NUM_FRAMES}")

    #Python
    print("Mierzę Python (NumPy)...")
    py_time, spec_py = compute_python_stft_time("input_signal.bin")
    spec_py.astype(np.float32).tofile(Path(outdir) / "spec_python.bin")

    #OpenMP
    omp_times = []
    spec_omp_saved = None
    if not Path(BIN_OMP).exists():
        print(f"Uwaga: {BIN_OMP} nie istnieje. Pomijam testy OpenMP.")
        omp_times = [0.0] * len(OMP_THREADS)
    else:
        for t in OMP_THREADS:
            print(f"Mierzę OpenMP ({t} threads)...")
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(t)
            t_exec = run_cpp_program(BIN_OMP, env=env)
            omp_times.append(t_exec)

            if Path("output_omp.bin").exists():
                try:
                    spec_omp = np.fromfile("output_omp.bin", dtype=np.float64)
                    expected = NUM_FRAMES * SPECTRUM_SIZE
                    if spec_omp.size != expected:
                        spec_omp = spec_omp[:expected]
                    spec_omp = spec_omp.reshape((NUM_FRAMES, SPECTRUM_SIZE)).T
                    spec_omp.astype(np.float32).tofile(Path(outdir) / f"spec_omp_t{t}.bin")
                    if spec_omp_saved is None:
                        spec_omp_saved = spec_omp
                except Exception as e:
                    print("Błąd przy wczytywaniu output_omp.bin:", e)


    #CUDA
    cuda_time = 0.0
    spec_cuda_saved = None
    if not Path(BIN_CUDA).exists():
        print(f"Uwaga: {BIN_CUDA} nie istnieje. Pomijam testy CUDA.")
    else:
        print("Mierzę CUDA...")
        cuda_time = run_cpp_program(BIN_CUDA)
        if Path("output_cuda.bin").exists():
            try:
                spec_cuda = np.fromfile("output_cuda.bin", dtype=np.float32)
                expected = NUM_FRAMES * SPECTRUM_SIZE
                if spec_cuda.size != expected:
                    spec_cuda = spec_cuda[:expected]
                spec_cuda = spec_cuda.reshape((NUM_FRAMES, SPECTRUM_SIZE)).T
                spec_cuda.astype(np.float32).tofile(Path(outdir) / "spec_cuda.bin")
                spec_cuda_saved = spec_cuda
            except Exception as e:
                print("Błąd przy wczytywaniu output_cuda.bin:", e)


    #wykresy + wyniki
    plot_performance(song_label, duration_s, py_time, omp_times, cuda_time, OMP_THREADS, outdir)
    plot_validation_spectrograms(song_label, spec_py, spec_omp_saved, spec_cuda_saved, outdir)

    res = {
        "song": song_label,
        "samples": SIGNAL_LEN,
        "duration_s": duration_s,
        "python_time": py_time,
        "cuda_time": cuda_time
    }
    for i, t in enumerate(OMP_THREADS):
        res[f"omp_t{t}"] = omp_times[i] if i < len(omp_times) else 0.0


    omp_nonzero = [v for v in [res[f"omp_t{t}"] for t in OMP_THREADS] if v > 0]
    best_omp = min(omp_nonzero) if omp_nonzero else 0.0
    res["best_omp_time"] = best_omp
    res["speedup_cuda_vs_python"] = (res["python_time"] / res["cuda_time"]) if res["cuda_time"]>0 else 0.0
    res["speedup_bestomp_vs_python"] = (res["python_time"] / best_omp) if best_omp>0 else 0.0

    return res





def main():
    results = []
    for song in SONGS:
        pass

if __name__ == "__main__":
    results = []
    for song in SONGS:
        if not Path(song).exists():
            print(f"Plik nie istnieje: {song} -> pomijam")
            continue
        label = Path(song).stem.replace(" ", "_")
        out_folder = OUT_DIR / label
        print(f"Benchmark dla: {song}")
        r = benchmark_one_song(song, label, out_folder)
        if r:
            results.append(r)

    #zapis CSV
    if results:
        fieldnames = ["song","samples","duration_s","python_time","cuda_time","best_omp_time","speedup_cuda_vs_python","speedup_bestomp_vs_python"] + [f"omp_t{t}" for t in OMP_THREADS]
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Wyniki zapisane do: {CSV_FILE}")

        #podsumowanie: wykres porownawczy najlepszych czasów
        songs = [r["song"] for r in results]
        py_times = [r["python_time"] for r in results]
        cuda_times = [r["cuda_time"] for r in results]
        best_omp_times = [r["best_omp_time"] for r in results]

        x = np.arange(len(songs))
        width = 0.25
        plt.figure(figsize=(10,6))
        p1 = plt.bar(x - width, py_times, width, label="Python")
        p2 = plt.bar(x, best_omp_times, width, label="Best OMP")
        p3 = plt.bar(x + width, cuda_times, width, label="CUDA")
        plt.yscale("log")
        plt.xticks(x, songs)
        plt.ylabel("Czas [s] (log)")
        plt.title("Porównanie czasów: Python / najlepsze OMP / CUDA")
        plt.legend()
        for bars in (p1,p2,p3):
            for bar in bars:
                h = bar.get_height()
                if h>0:
                    plt.text(bar.get_x()+bar.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        combo_path = OUT_DIR / "summary_comparison.png"
        plt.tight_layout()
        plt.savefig(combo_path)
        plt.close()
        print(f"Zapisano podsumowanie: {combo_path}")
    else:
        print("Brak wyników do zapisania.")
