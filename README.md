# Speech to Text RNN

A Recurrent Neural Network (RNN) that is trained on the LibriSpeech (currently LibriLightLimited) dataset.


# Table of Contents

- [Installation](#installation)
- [Usage](#usage)
	- [Display Sample Data](#display-sample-data)
	- [Preprocess Audio Data](#preprocess-audio-data)
	- [Read Preprocessed MFCCs](#read-preprocessed-mfccs)


# Installation

Requires Python 3.7 or later.

## Install Dependencies

```bash
cd speech_to_text
pip install -r requirements.txt
```


# Usage

## Display Sample Data

Dataset will be downloaded automatically into "data" folder in project root directory.

```bash
python train.py --display

# Additional flag options
--waveform  # Show waveform
--play  # Play audio
--spectrogram  # Show spectrogram
--mfcc  # Show MFCCs

# Example: Show test data, display waveform, and play audio
python train.py --display --waveform --play
```

## Preprocess Audio Data

Extract MFCC data using Librosa and store with selected features using h5py.

```bash
python train.py --preprocess
```

## Read Preprocessed MFCCs

Will display MFCC shape and extracted features for each file.

```bash
python train.py --read-mfcc
```