# Author: Hyungshin Ryu
# Contact: rhss10@snu.ac.kr
# convert audio and transcription path to csv files

import json
import os

import librosa

# Change TRANSCRIPTION_PATH and AUDIO_PATH before executing the code
TRANSCRIPTION_PATH = "/workspace/data/speech_corpus/NIA-10/nia-10-100per/json/jp/"
AUDIO_PATH = "/workspace/data/speech_corpus/NIA-10/nia-10-100per/audio/jp/"
data_file = open("./nia-10_17less.txt", "w")
extra_file = open("./nia-10_17more.txt", "w")
invalid_file = open("./nia-10_invalid.txt", "w")

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

            audio = (
                AUDIO_PATH + r.split("/jp/")[1] + "/" + file.split(".json")[0] + ".wav"
            )
            if os.path.exists(audio) and text != "":
                audio_duration = librosa.get_duration(filename=audio)
                if audio_duration < 17.0:
                    data_file.write(audio + "\t" + text + "\n")
                else:
                    extra_file.write(audio + "\t" + text + "\n")
            else:
                invalid_file.write(os.path.join(r, file) + "\t" + text + "\n")


data_file.close()
extra_file.close()
invalid_file.close()
print("- Data extraction done.")
