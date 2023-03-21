## General
- A source code to fine-tune self-supervised learning model (SSL) on NIA-2022-2-10 Japanese Dataset for Japanese Automatic Speech Recognition (ASR) of total 1028 hrs.
- NIA-2022-2-10 Japanese Dataset for Japanese Automatic Speech Recognition (ASR) will soon be released within 2023.
- More information regarding the **usage of the dataset** and **docker support** will be updated with the relase of dataset.
- If you're looking at this file through Docker ver 3.0, you may want to refer to https://github.com/rhss10/japanese_asr_nia-22-2-10 for the latest codes.

## Setup
- The code splits the dataset into 8:1:1 train, valid, test + (files over 17 second) set for efficient GPU usage.
- The train code utilizes the basic Trainer API by Huggingface. By default, the SSL model is set to Wav2Vec2-xls-r-300m.

## Performance
- Valid WER, CER: 4.50%, 2.33%,
- Test WER, CER: 5.12%, 2.62%

## Commands
### Prepare Data
```python
# Data processing should be done beforehand on the ACTUAL data path
python extract_data.py
python create_datasets.py
```
### Train
```python
# Example command for training. Refer to train.py for more supported arguments
python train.py --exp_prefix NIA-10
```
### Test
```python
# Example
python test.py
```
