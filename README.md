# Speech to Text

A transformer based neural network designed for Speech to Text (S2T) and Automatic Speech Recognition (ASR) tasks. Takes in human speech audio and outputs readable text in the target language.

Current available langauges: English


# Table of Contents

- [Installation & Usage](#installation--usage)
	- [Install Dependencies](#install-dependencies)
	- [Script Arguments](#script-arguments)
	- [Preprocessing](#preprocessing)
	- [Training](#training)
	- [Evaluation](#evaluation)
- [Dataset](#dataset)
- [Word Embeddings](#word-embeddings)
- [References](#references)


# Installation & Usage

Requires Python 3.7 or later.

## Install Dependencies

```bash
cd speech_to_text
pip install -r requirements.txt
```

## Script Arguments

To see all available options for each module script, use the argparse help flag:

```bash
python train.py --help
```

## Preprocessing

To begin training, the MFCC data needs to be derived from the raw audio samples. To do this preprocessing,
run the preprocess module script for each split:

```bash
python preprocess.py -s [split_name]
```

This will download the selected dataset split (default is "clean-dev") into the `data` folder, and extract the MFCC
data to the `mfcc` folder, which will be used in training. Alternatively, you can run the train module directly, and
the preprocessing will begin automatically.

A custom dataset can be placed in the `data` folder in lieu of the provided splits, **as long as the split is added to the `splits.py` class.**

You can also check out samples from the dataset, such as playing audio files and displaying spectrogram data. See the help
flag for more details.


## Training

To begin training, run the train module script:

```bash
python train.py
```

The model settings can be adjusted from the `config.py` file. Throughout training, tensorboard metric data will be periodically
outputted to the `runs` folder. Once training is complete, the model parameters, vocabulary dictionary, and optimizer will be
saved to the `models` folder.


## Evaluation

To evaluate a trained model on custom audio files, run the evaluate module script:

```bash
python evaluate.py -m [model] -f [directory_containing_audio_files]
```

This will automatically preprecess the audio files and output the model's prediction.


# Dataset

The model's training regimen incorporates data from the following public datasets:

- [CommonVoice Corpus 20.0](https://commonvoice.mozilla.org/en/datasets)
- [LibriSpeech](https://pytorch.org/audio/stable/generated/torchaudio.datasets.LIBRISPEECH.html)


# Word Embeddings

This model has not incorporated any pre-trained vectorized word embeddings. They have been randomly initalized and learned directly from the training datasets.

Pre-trained embeddings can be loaded via PyTorch's `nn.Embedding` class inside the `WordEmbedder` module.


# References

The following resources were used as inspiration for the neural network's architecture.

- [Attention Is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), the original paper which has been cited [over 90,000 times](https://scholar.google.com/scholar?lr&ie=UTF-8&oe=UTF-8&q=Attention+is+All+You+Need+Vaswani+Shazeer+Parmar+Uszkoreit+Jones+Gomez+Kaiser+Polosukhin)
	- [Annotated version (Harvard)](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [fairseq (Facebook Research)]()
	- [wav2vec 2.0 - Paper](https://paperswithcode.com/paper/wav2vec-2-0-a-framework-for-self-supervised)
	- [wav2vec 2.0 - Example](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md)
	- [wav2vec 2.0 - Model](https://github.com/facebookresearch/fairseq/tree/main/fairseq/models/wav2vec)
	- [Speech-to-Text - Example](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/README.md)
	- [Speech-to-Text - Model](https://github.com/facebookresearch/fairseq/tree/main/fairseq/models/speech_to_text)
- [Whisper (OpenAI)](https://github.com/openai/whisper)
	- [Architecture](https://openai.com/research/whisper#fn-4)