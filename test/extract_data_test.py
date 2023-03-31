# Author: Hyungshin Ryu
# Contact: rhss10@snu.ac.kr
# convert audio and transcription path to csv files

import json
import os

# Change TRANSCRIPTION_PATH and AUDIO_PATH before executing the code
TRANSCRIPTION_PATH = "/workspace/data/NIA-10-backup/json/"
AUDIO_PATH = "/workspace/data/NIA-10-backup/audio/"
data_file = open("./nia-10_inference.txt", "w")

for r, d, f in os.walk(TRANSCRIPTION_PATH):
    d.sort()
    f.sort()
    for file in f:
        if ".json" in file:
            with open(os.path.join(r, file), "r", encoding="utf-8-sig") as json_file:
                dic = json.load(json_file)
                text = dic["dialogs"]["text"]
                text = text.replace("\n", "")
                text = text.replace("\r", "")
                text = text.replace("\t", " ")

            audio = AUDIO_PATH + file.split(".json")[0] + ".wav"
            data_file.write(audio + "\t" + text + "\n")


data_file.close()
print("- Inference data extraction done.")
