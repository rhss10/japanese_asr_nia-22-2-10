## General
- A source code to fine-tune Wav2Vec2-xls-r on NIA-2022-2-10 Japanese Dataset 

## Commands
### Prepare Data
```python
# Data processing should be done beforehand on the ACTUAL data path
# ALL the data/* files including nia-10_train/, etc. will not work for your directory!
python extract_data.py
python create_datasets.py
```
### Train
```python
# Example command for training. Refer to train.py for more arguments
python train.py --exp_prefix NIA-10
```
### Test
```python
# Example
python test.py
```
