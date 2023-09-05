# Speech to Text RNN

A Recurrent Neural Network (RNN) that is trained on a large dataset of human speech.

Step 1: Classify human speech phenomes

Step 2: Identify words to form a sentence from audio file

Step 3: Use trained model to read live audio buffer and do live STT (Speech-To-Text)


# Usage and Installation

## Install Dependencies

Uses Python 3.7 or later.

```bash
cd speech_to_text
pip install -r requirements.txt
```

## Usage

Show Test Data
```bash
python train.py --show-data
```

Additional Flag Options

```bash
--waveform  # Show waveform
--play  # Play audio
--spectrogram  # Show spectrogram
--mfcc  # Show MFCCs
```

Example: Show test data, display waveform, and play audio
```bash
python train.py --show-data --waveform --play
```

Dataset will be downloaded automatically into "data" folder in project root directory.

(Note: Need to find smaller data subset.)