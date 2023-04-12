from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
from skimage.feature import peak_local_max
from config import config

sample_rate = config["sample_rate"]
hop_length = config["hop_length"]
n_fft = config["n_fft"]
song_length = config["song_length"]
k = config["k"]
feat = config["feat"]
min_distance = config["min_distance"]
threshold_abs = config["threshold_abs"]

max_time = sample_rate * song_length // hop_length


def audioIdentification(query_dir, fingerprint_dir, output_file):
    # Get all the fingerprints from the database
    fingerprints = list(Path(fingerprint_dir).glob("**/*.npy"))
    fingerprint_dicts = []
    output = ""

    for fingerprint in fingerprints:
        # Load fingerprint
        Ld = np.load(fingerprint, allow_pickle=True).item()
        fingerprint_dicts.append(Ld)

    # Get all the query files
    query_files = list(Path(query_dir).glob("**/*.wav"))

    for query_file in tqdm(query_files, total=len(query_files)):
        # Get the query fingerprints
        # Load audio file
        y, sr = librosa.load(query_file, sr=sample_rate)

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

        # Calculate matching function for each fingerprint
        max_vals = {}
        for fingerprint, fd in zip(fingerprints, fingerprint_dicts):
            indicators = create_indicator_functions(local_max, fd)
            matching_function = np.sum(indicators, axis=0)
            max_val = np.max(matching_function)
            max_vals[fingerprint.stem] = max_val

        max_vals = sorted(max_vals.items(), key=lambda x: x[1], reverse=True)

        # Save the top k results
        output += f"{query_file.stem}\t"
        for i in range(k):
            output += f"{max_vals[i][0]}\t"
        output += "\n"

    with open(output_file, "w") as f:
        f.write(output)


def create_indicator_functions(coords, Ld):
    indicator_functions = []
    for i in range(len(coords)):
        # set time zero is at -max_time
        indicator_function = np.zeros(max_time * 2 + 1)
        if coords[i][0] in Ld:
            # Found match
            for ts in Ld[coords[i][0]]:
                indicator_function[ts - coords[i][1] + max_time] = 1
        indicator_functions.append(indicator_function)
    return np.array(indicator_functions)
