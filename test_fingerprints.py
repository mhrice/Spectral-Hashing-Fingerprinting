from create_fingerprints import fingerprintBuilder
from audio_identification import audioIdentification
from evaluate import evaluate_results

print("Building fingerprints...")
fingerprintBuilder("data/database_recordings", "data/fingerprints")
print("Identifying audio...")
audioIdentification("data/query_recordings", "data/fingerprints", "data/output.txt")
evaluate_results("data/output.txt")
