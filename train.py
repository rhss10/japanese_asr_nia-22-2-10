from datasets import load_dataset, load_metric, Dataset, Audio, load_from_disk, concatenate_datasets
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from addict import Dict as addict
import argparse
import os, logging
from torch.utils.tensorboard import SummaryWriter


def prepare_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the model."
        ),
    )
    parser.add_argument(
        "--per_device_batch_size", type=int, default=4,
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="facebook/wav2vec2-xls-r-300m",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default='linear',
    )
    parser.add_argument(
        "--ctc_loss_reduction", type=str, default='mean',
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1,
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=30,
    )
    parser.add_argument(
        "--metric_for_best_model", type=str, default="eval_wer"
    )
    parser.add_argument(
        "--greater_is_better", action='store_true',
    )
    parser.add_argument(
        "--exp_prefix", type=str, default='',
        help="Custom string to add to the experiment name."
    )
    parser.add_argument(
        "--train_feature_extractor", action='store_true',
        help="Train convolution models in wav2vec2.",
    )

    args = parser.parse_args()
    args.exp_name = f"{args.exp_prefix}_bat{args.per_device_batch_size}_lr{args.learning_rate}_warm{args.warmup_ratio}"
    args.save_dir_path = './models/' + args.exp_name
    args.save_log_path = './logs/tb_tracker/' + args.exp_name
    os.makedirs(args.save_dir_path, exist_ok=False)
    os.makedirs(args.save_log_path, exist_ok=False)

    return args


def prepare_dataset(batch):
    array = batch['audio']['array']
    if array.dtype != 'float32':
        array = np.float32(array)
        
    # batched output is "un-batched"
    batch["input_values"] = processor(array, sampling_rate=16000).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["ans"]).input_ids

    return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {'wer': wer, 'cer': cer}


def prepare_trainer(args, processor, train_ds, test_ds):
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name_or_path,
        ctc_loss_reduction=args.ctc_loss_reduction, 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    if not args.train_feature_extractor:
        print('- Freezing feature extractor')
        model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=args.save_dir_path,
        logging_dir=args.save_log_path,
        report_to=['tensorboard'],
        group_by_length=False,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        log_level='info',
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=3,
        push_to_hub=False,
        load_best_model_at_end=True,
        greater_is_better=args.greater_is_better,
        metric_for_best_model=args.metric_for_best_model,
        dataloader_num_workers=15,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    return trainer
    

if __name__ == '__main__':
    args = prepare_arguments()
    logger = logging.getLogger(__name__)

    TRAIN_DS = './data/nia-10_train'
    VALID_DS = './data/nia-10_valid_small'
    VOCAB = './vocab.json'
    print(TRAIN_DS, VALID_DS, VOCAB, sep='\n')
    ds_train = load_from_disk(TRAIN_DS)
    ds_valid = load_from_disk(VALID_DS)
    tokenizer = Wav2Vec2CTCTokenizer(VOCAB, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                        return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(args.save_dir_path)

    ds_train = ds_train.map(prepare_dataset, num_proc=15)
    ds_valid = ds_valid.map(prepare_dataset, num_proc=15)

    cer_metric = load_metric("cer")
    wer_metric = load_metric("wer")

    trainer = prepare_trainer(args, processor, train_ds=ds_train, test_ds=ds_valid)
    trainer.train()
    trainer.save_state()
    trainer.save_model()
    metrics = trainer.evaluate()
    trainer.save_metrics('all', metrics)

    print("- Training complete.")
