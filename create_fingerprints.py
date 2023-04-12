from pathlib import Path
import librosa
import numpy as np
from skimage.feature import peak_local_max
from tqdm import tqdm
from config import config

sample_rate = config["sample_rate"]
hop_length = config["hop_length"]
n_fft = config["n_fft"]
song_length = config["song_length"]
min_distance = config["min_distance"]
threshold_abs = config["threshold_abs"]
feat = config["feat"]


def fingerprintBuilder(database_path, fingerprints_path):
    # Get all files in database
    files = list(Path(database_path).glob("**/*.wav"))

    # Create output directory
    Path(fingerprints_path).mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(files, total=len(files)):
        # Load audio file
        y, sr = librosa.load(file_path, sr=sample_rate)
        # Pad or trim to 30 seconds
        if len(y) < sample_rate * song_length:
            y = np.pad(y, (0, sample_rate * song_length - len(y)), "constant")
        else:
            y = y[: sample_rate * song_length]

        if feat == "chroma":
            feature = librosa.feature.chroma_stft(
                y=y, sr=sr, hop_length=hop_length, n_fft=n_fft
            )
        elif feat == "mfcc":
            feature = librosa.feature.mfcc(
                y=y, sr=sr, hop_length=hop_length, n_fft=n_fft
            )
        elif feat == "mel":
            feature = librosa.feature.melspectrogram(
                y=y, sr=sr, hop_length=hop_length, n_fft=n_fft
            )
        elif feat == "stft":
            feature = np.abs(
                librosa.stft(y=y, hop_length=hop_length, n_fft=n_fft, center=False)
            )
        # Compute local maxima
        local_max = peak_local_max(
            feature, min_distance=min_distance, threshold_abs=threshold_abs
        )

        # Create inverted list
        il = create_inverted_list(local_max)
        # Save fingerprints
        output_path = Path(fingerprints_path) / (file_path.stem + ".npy")
        np.save(output_path, il)


def create_inverted_list(coords):
    Ld = {}
    for i in range(len(coords)):
        if coords[i][0] not in Ld:
            Ld[coords[i][0]] = []
        Ld[coords[i][0]].append(coords[i][1])
    return Ld
