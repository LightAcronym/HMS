
# Harmful Brain Activity Classification
### Project Overview
This project is part of the Kaggle competition "HMS Harmful Brain Activity Classification." The goal of this competition
is to develop machine learning models that can identify harmful brain activity patterns from EEG data.
This challenge aims to push forward the field of neuroinformatics and contribute to health informatics by automating the
detection of harmful brain activity, potentially improving patient outcomes in clinical settings."

### Competition Data
The dataset provided by the competition consists of EEG and spectrogram recordings from multiple subjects. 
For each recording, a team of expert neurologists cast votes for one of six possible diagnoses -- Seizure, LPD, GPD, 
LRDA, GRDA, or Other -- based on the presence of specific brain activity patterns present in the spectrograms.
Participants are expected to use this data to train models capable of predicting the normalized vote distribution
for each recording.

### Dependencies

- Python 3.11
- Other dependencies can be found in the __requirements.txt__ file

### Usage

1. Clone the repository or download the following scripts:

- `images` (directory)
- `inference.py`
- `DataGenerator.py`
- `config.json`
- `model_weights.h5`
- `589979261.parquet` (for testing the application)

2. Follow requirements.txt installations

3. Run the script in your cmd:

`streamlit run inference.py`

Example parquet called: 589979261.parquet at 'data' folder

### Acknowledgements
Thank you to Kaggle and HMS for hosting the competition and providing the dataset, and to the ITC staff for guidance and mentorship during this project.

### Authors
Or Gewelber ([LinkedIn](https://www.linkedin.com/in/or-gewelber/)) ([Github](https://github.com/LightAcronym))

Zoe Stankowski ([LinkedIn](https://www.linkedin.com/in/zoe-stankowska/)) ([Github](https://github.com/zstankow))

Sacha Koskas ([LinkedIn](https://www.linkedin.com/in/sacha-koskas-a3a46b1b5/)) ([Github](https://github.com/SachaKsk))
