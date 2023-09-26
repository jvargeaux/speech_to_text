# Speech to Text RNN

A Recurrent Neural Network (RNN) that is trained on the LibriSpeech (currently LibriLightLimited) dataset.


# References

## Transformer Architecture

- [Attention Is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), the original paper which has been cited [over 90,000 times](https://scholar.google.com/scholar?lr&ie=UTF-8&oe=UTF-8&q=Attention+is+All+You+Need+Vaswani+Shazeer+Parmar+Uszkoreit+Jones+Gomez+Kaiser+Polosukhin)
	- [Annotated version (Harvard)](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

## Automatic Speech Recognition (ASR) Implementation

- [fairseq (Facebook Research)]()
	- [wav2vec 2.0 - Paper](https://paperswithcode.com/paper/wav2vec-2-0-a-framework-for-self-supervised)
	- [wav2vec 2.0 - Example](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md)
	- [wav2vec 2.0 - Model](https://github.com/facebookresearch/fairseq/tree/main/fairseq/models/wav2vec)
	- [Speech-to-Text - Example](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/README.md)
	- [Speech-to-Text - Model](https://github.com/facebookresearch/fairseq/tree/main/fairseq/models/speech_to_text)


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