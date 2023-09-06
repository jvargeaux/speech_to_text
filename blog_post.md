	This is just an outline. It is unfinished.

# Audio Data Preprocessing for Deep Learning

Speech-To-Text Recurrent Neural Network

## Dataset

The dataset used is [LibraLightLimited](https://pytorch.org/audio/stable/generated/torchaudio.datasets.LibriLightLimited.html#torchaudio.datasets.LibriLightLimited), which as a subset of LibriSpeech is still quite large. For testing purposes, I'll be using a random subset of LibriLightLimited to further reduce size and processing time.


## Brief Background on Digital Audio Terms & Concepts

Sample Rate (sampling)

Bit Depth (quantization)

Frequency (Nyquist)

Fourier Transform

Short Time Fourier Transform (STFT)

Time vs Frequency domains

Spectrogram

Mel Frequency .. Coefficient (MFCC)


## The Dataset


## Recurrent Neural Network Architecture

Input shape

Output shape


### Layers
- LSTM Layer (64)
- LSTM Layer (64)
- Dense Layer
- Dropout Layer (to prevent overfitting)
- Dense Layer (softmax output)


## Preprocessing Audio Data

### Flow

1. Take random subset of data (for reducing processing)
2. For each audio file in subset:
	- Extract samples & sample rate
	- Extract MFCCs by performing STFTs across all samples
	- Save selected features along with extracted MFCCs
3. Export all data features to json or csv file

We will use this exported file containing the relevant features and MFCC data to train our model


### Why MFCC instead of spectrogram (STFT data) or raw samples?