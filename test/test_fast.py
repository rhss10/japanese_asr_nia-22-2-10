# Author: Hyungshin Ryu
# Contact: rhss10@snu.ac.kr
# test the performance of the best checkpoint (default batch size 32, thus faster than ../test.py)

import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


# collator
def collate_fn(batch):
    return {
        "input_values": processor(
            [np.float32(x["audio"]["array"]) for x in batch],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_values,
        "ans": [x["ans"] for x in batch],
        "path": [x["audio"]["path"] for x in batch],
    }


# evaluation
def test(dataloader):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    for x in tqdm(dataloader):
        with torch.no_grad():
            logits = model(
                x["input_values"].to(device),
            ).logits
        pred_ids = torch.argmax(logits, dim=-1)
        preds = processor.batch_decode(pred_ids)
        wer_metric.add_batch(predictions=preds, references=x["ans"])
        cer_metric.add_batch(predictions=preds, references=x["ans"])

    print("TEST WER: {:2f}%".format(100 * wer_metric.compute()))
    print("TEST CER: {:2f}%".format(100 * cer_metric.compute()))


# load data, processor and model
TEST = "./nia-10_infer/"
ds = load_from_disk(TEST)
BATCH_SIZE = 32
test_dataloader = torch.utils.data.DataLoader(
    ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, pin_memory=True
)
PTM = "../models/NIA_bat16_lr0.0001_warm0.1"
device = "cuda:1"
processor = Wav2Vec2Processor.from_pretrained(PTM)
model = Wav2Vec2ForCTC.from_pretrained(PTM)
model.to(device)

print("PTM:", PTM)
print("TESTSET:", TEST)
print("BATCHSIZE:", BATCH_SIZE)
test(test_dataloader)
print("- Test finished.")
