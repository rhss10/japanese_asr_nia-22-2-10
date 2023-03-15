from datasets import load_from_disk, Audio, concatenate_datasets, Dataset, load_dataset, load_metric
import json
import re
import unicodedata
import pandas as pd
import MeCab
import pykakasi


# set parser
wakati = MeCab.Tagger("-Owakati")
kakasi = pykakasi.kakasi()

#NOTE: remove only the special characters and leave alphabets (hiragana, english, russian, french)
CHARS_TO_IGNORE = [",", "．", "・", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", "@", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "'", "ʻ", "ˆ",
                   "｢", "｣", '‐', '―', '′', '⁻', '∙', '\u3000', '-', '&', '|', 'ᐨ', '∼', '≪', '≫', '×', chr(12442), "⋅", "☆"]
#CHARS_TO_IGNORE_NIA = []
chars_to_remove_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}가-힣ㄱ-ㅎㅏ-ㅣ\u1100-\u11ff]"
#chars_to_remove_regex_nia10 = f"[{re.escape(''.join(CHARS_TO_IGNORE_NIA))}]"


def parse_remove_special_characters_for_nia10(batch):
    batch["ans"] = re.sub('\(.+?\)', '', batch["text"])
    batch["ans"] = unicodedata.normalize('NFKC', batch["ans"])
    batch["ans"] = wakati.parse(batch["ans"]).strip()
    batch["ans"] = re.sub(chars_to_remove_regex, '', batch["ans"]).lower().strip()
    #batch["ans"] = re.sub(chars_to_remove_regex_nia10, '', batch["ans"])
    batch["ans"] = re.sub(' +', ' ', batch["ans"])

    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["ans"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


# make datasets
nia_less = load_dataset('csv', data_files='./nia-10_17less.txt', delimiter='\t', column_names=['audio', 'text'], split='train')
nia_more = load_dataset('csv', data_files='./nia-10_17more.txt', delimiter='\t', column_names=['audio', 'text'], split='train')
nia_less = nia_less.cast_column('audio', Audio(sampling_rate=16000))
nia_more = nia_more.cast_column('audio', Audio(sampling_rate=16000))
nia_less = nia_less.map(parse_remove_special_characters_for_nia10)
nia_more = nia_more.map(parse_remove_special_characters_for_nia10)

print(f'RAW TRAIN LEN:{len(nia_less)}, RAW TEST LEN: {len(nia_more)}')
nia_less = nia_less.filter(lambda x: x['ans'] != '', num_proc=15)
nia_more = nia_more.filter(lambda x: x['ans'] != '', num_proc=15)
print(f'FINAL TRAIN LEN:{len(nia_less)}, FINAL TEST LEN: {len(nia_more)}')

nia_ds = nia_less.train_test_split(test_size=0.2)
nia_test = nia_ds['test'].train_test_split(test_size=0.5)

# make vocabularies
nia_vocab = nia_ds['train'].map(extract_all_chars, batched=True, batch_size=-1, remove_columns=nia_ds['train'].column_names)
vocab_list = list(set(nia_vocab["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open('../vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)


nia_more.save_to_disk('./nia-10_long', num_proc=15, max_shard_size="1GB")
nia_ds['train'].save_to_disk('./nia-10_train', num_proc=15, max_shard_size="1GB")
nia_test['train'].save_to_disk('./nia-10_valid', num_proc=15, max_shard_size="1GB")
nia_small = nia_test['train'].train_test_split(test_size=0.1)
nia_small['test'].save_to_disk('./nia-10_valid_small')
nia_test['test'].save_to_disk('./nia-10_test', num_proc=15, max_shard_size="1GB")
print("- Finished making datasets.")
