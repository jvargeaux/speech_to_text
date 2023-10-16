# Speech to Text

A transformer based neural network designed for Automatic Speech Recognition (ASR). Takes in human speech audio
and outputs readable text in the target language.

Initial training done on the LibriSpeech dataset.


# Table of Contents

- [Installation & Usage](#installation--usage)
	- [Install Dependencies](#install-dependencies)
	- [Script Arguments](#script-arguments)
	- [Preprocessing](#preprocessing)
	- [Training](#training)
	- [Evaluation](#evaluation)
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
run the preprocess module script:

```bash
python preprocess.py
```

This will download the selected dataset split (default is "clean-dev") into the `data` folder, and extract the MFCC
data to the `mfcc` folder, which will be used in training. Alternatively, you can run the train module directly, and
the preprocessing will begin automatically.

A custom dataset can be placed in the `data` folder in lieu of a LibriSpeech split.

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
python evaluate.py [file paths]
```

This will automatically preprecess the audio files and output the model's prediction.


# References

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