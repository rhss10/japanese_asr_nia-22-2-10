# Author: Hyungshin Ryu
# Contact: rhss10@snu.ac.kr
# test the best model on valid, test set

import evaluate
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# load data, processor and model
VALID = "./data/nia-10_valid/"
TEST = "./data/nia-10_test/"
LONG = "./data/nia-10_long/"
valid_dataset = load_from_disk(VALID)
test_dataset = load_from_disk(TEST)
long_dataset = load_from_disk(LONG)

PTM = "./models/NIA_bat16_lr0.0001_warm0.1"
print("PTM:", PTM)
print("VALIDSET", VALID)
print("TESTSET:", TEST)
print("LONGSET:", LONG)
device = "cuda"
processor = Wav2Vec2Processor.from_pretrained(PTM)
model = Wav2Vec2ForCTC.from_pretrained(PTM)
model.to(device)


# evaluation
def prepare_dataset(batch):
    array = batch["audio"]["array"]
    inputs = processor(array, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(
            inputs.input_values.to(device),
            attention_mask=inputs.attention_mask.to(device),
        ).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    preds = processor.decode(pred_ids)
    wer_metric.add(prediction=preds, reference=batch["ans"])
    cer_metric.add(prediction=preds, reference=batch["ans"])

    return batch


# load evaluation metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# evaluate on valid set
valid_dataset = valid_dataset.map(prepare_dataset, batch_size=16)
print("VALID WER: {:2f}%".format(100 * wer_metric.compute()))
print("VALID CER: {:2f}%".format(100 * cer_metric.compute()))

# evaluate on test set
test_dataset = test_dataset.map(prepare_dataset, batch_size=16)
long_dataset = long_dataset.map(prepare_dataset)
print("TEST WER: {:2f}%".format(100 * wer_metric.compute()))
print("TEST CER: {:2f}%".format(100 * cer_metric.compute()))
print("- Test finished.")
