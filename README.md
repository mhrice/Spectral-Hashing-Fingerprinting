# Installation
All packages have been tested in the lab environment.
However, if there is some issue, run `pip install -r requirements.txt` to install all the packages.

# Download train data
```
mkdir data
cd data
wget https://collect.qmul.ac.uk/down?t=5DSGJQ07VCPS2S4C/613IT8K160JFB9U6AILKEPO
unzip down\?t\=5DSGJQ07VCPS2S4C%2F613IT8K160JFB9U6AILKEPO
wget https://collect.qmul.ac.uk/down?t=65A2P9AQ9276OBA9/6L0T7NQ4ULSHHH13758JR6O
unzip down\?t\=65A2P9AQ9276OBA9%2F6L0T7NQ4ULSHHH13758JR6O
```

# Run train and evaluation
`python test_fingerprints.py`

# Use script for creating and testing fingerprints.
Based on `test_fingerprints.py`

```
from create_fingerprints import fingerprintBuilder
from audio_identification import audioIdentification

fingerprintBuilder("data/database_recordings", "data/fingerprints")
audioIdentification("data/query_recordings", "data/fingerprints", "data/output.txt")
```

