## General
- A source code to fine-tune self-supervised learning model (SSL) on NIA-2022-2-10 Japanese Dataset for Japanese Automatic Speech Recognition (ASR) of total 1028 hrs.
- NIA-2022-2-10 Japanese Dataset for Japanese Automatic Speech Recognition (ASR) will soon be released within 2023.
- More information regarding the **usage of the dataset** and **docker support** will be updated with the relase of dataset.
- If you're looking at this file through Docker ver 3.0, you may want to refer to https://github.com/rhss10/japanese_asr_nia-22-2-10 for the latest codes.

## License
- SPDX-FileCopyrightText: © 2023 Hyungshin Ryu \<rhss10@snu.ac.kr\>
- SPDX-License-Identifier: Apache-2.0

## Setup
- The code splits the dataset into 8:1:1 train, valid, test + (files over 17 second) set for efficient GPU usage.
- The train code utilizes the basic Trainer API by Huggingface. By default, the SSL model is set to Wav2Vec2-xls-r-300m.

## Performance
- Valid WER, CER: 4.50%, 2.33%,
- Test WER, CER: 5.12%, 2.62%

## For Test-only (TTA Qualification/Docker)
### Notes
- This section is used for
    1. TTA Qualification
    2. those who want to test the performance of the best model checkpoint using Docker (model checkpoint included inside)
- With audio and json directory path provided, the test script will evaluate the WER/CER of the model checkpoint
- The example test split list is shown in **test_final.txt** 

### Command
```bash
cd test
# Don't forget to change DIRECTORY PATH inside test/extract_data_test.py before executing the bash script
sh test.sh
# Done!
```

## For Train/Test
### 1. Prepare Data
```python
# Data processing should be done beforehand on the ACTUAL data path
# The example files (data/nia-10*.txt) will not work!
python extract_data.py
python create_datasets.py
```
### 2. Train
```python
# Example command for training. Refer to train.py for more supported arguments
python train.py --exp_prefix NIA-10
```
### 3. Test
```python
# Example
python test.py
```
