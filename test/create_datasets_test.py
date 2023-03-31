# Author: Hyungshin Ryu
# Contact: rhss10@snu.ac.kr
# convert csv files into huggingface datasets format and create vocabulary set
# creates additional small valid set as the total valid set (100 hrs) will consume too much GPU

import json
import re
import unicodedata

import MeCab
import pykakasi
from datasets import (
    Audio,
    Dataset,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    load_metric,
)

# set parser
wakati = MeCab.Tagger("-Owakati")
kakasi = pykakasi.kakasi()

# NOTE: remove only the special characters and hangeuls (as they are used for phonetic transcription) and leave alphabets (hiragana, english, russian, french)
CHARS_TO_IGNORE = [
    ",",
    "．",
    "・",
    "?",
    "¿",
    ".",
    "!",
    "¡",
    ";",
    "；",
    ":",
    '""',
    "%",
    "@",
    '"',
    "�",
    "ʿ",
    "·",
    "჻",
    "~",
    "՞",
    "؟",
    "،",
    "।",
    "॥",
    "«",
    "»",
    "„",
    "“",
    "”",
    "「",
    "」",
    "‘",
    "’",
    "《",
    "》",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "=",
    "`",
    "_",
    "+",
    "<",
    ">",
    "…",
    "–",
    "°",
    "´",
    "ʾ",
    "‹",
    "›",
    "©",
    "®",
    "—",
    "→",
    "。",
    "、",
    "﹂",
    "﹁",
    "‧",
    "～",
    "﹏",
    "，",
    "｛",
    "｝",
    "（",
    "）",
    "［",
    "］",
    "【",
    "】",
    "‥",
    "〽",
    "『",
    "』",
    "〝",
    "〟",
    "⟨",
    "⟩",
    "〜",
    "：",
    "！",
    "？",
    "♪",
    "؛",
    "/",
    "\\",
    "º",
    "−",
    "^",
    "'",
    "ʻ",
    "ˆ",
    "｢",
    "｣",
    "‐",
    "―",
    "′",
    "⁻",
    "∙",
    "\u3000",
    "-",
    "&",
    "|",
    "ᐨ",
    "∼",
    "≪",
    "≫",
    "×",
    chr(12442),
    "⋅",
    "☆",
]
# CHARS_TO_IGNORE_NIA = []
# NOTE: \u1100-\u11ff refers to johap hangeuls
chars_to_remove_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}가-힣ㄱ-ㅎㅏ-ㅣ\u1100-\u11ff]"
# chars_to_remove_regex_nia10 = f"[{re.escape(''.join(CHARS_TO_IGNORE_NIA))}]"


def parse_remove_special_characters_for_nia10(batch):
    batch["ans"] = re.sub("\(.+?\)", "", batch["text"])
    batch["ans"] = unicodedata.normalize("NFKC", batch["ans"])
    batch["ans"] = wakati.parse(batch["ans"]).strip()
    batch["ans"] = re.sub(chars_to_remove_regex, "", batch["ans"]).lower().strip()
    # batch["ans"] = re.sub(chars_to_remove_regex_nia10, '', batch["ans"])
    batch["ans"] = re.sub(" +", " ", batch["ans"])

    return batch


# make datasets
nia_infer = load_dataset(
    "csv",
    data_files="./nia-10_inference.txt",
    delimiter="\t",
    column_names=["audio", "text"],
    split="train",
)

nia_infer = nia_infer.cast_column("audio", Audio(sampling_rate=16000))
nia_infer = nia_infer.map(parse_remove_special_characters_for_nia10)
nia_infer.save_to_disk("./nia-10_infer", num_proc=15, max_shard_size="5GB")
print("- Finished making inference datasets.")
