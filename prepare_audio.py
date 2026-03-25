import argparse
import librosa
import numpy as np
from pathlib import Path

#wczytanie audio, konwersja na mono -> zapis jako float32 binarny
def prepare_input(input_path, output_bin="input_signal.bin", target_sr=44100):
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {input_path}")
    print(f"Wczytuję plik: {input_path} (resample -> {target_sr} Hz, mono)")
    audio, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
    audio = audio.astype(np.float32)
    audio.tofile(output_bin)
    print(f"Zapisano: {output_bin}  (próbek: {len(audio)}, czas: {len(audio)/target_sr:.2f}s)")
    return len(audio), target_sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input_signal.bin from an audio file")
    parser.add_argument("--input", "-i", required=True, help="Wejściowy plik audio (mp3,wav...)")
    parser.add_argument("--output", "-o", default="input_signal.bin", help="Plik wyjściowy (raw float32)")
    parser.add_argument("--sr", type=int, default=44100, help="Docelowe próbkowanie")
    args = parser.parse_args()
    try:
        prepare_input(args.input, output_bin=args.output, target_sr=args.sr)
    except Exception as e:
        print("Błąd przygotowania pliku:", e)
        raise SystemExit(1)
