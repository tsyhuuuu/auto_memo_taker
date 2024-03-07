import os, time
import subprocess
from whisper import whisper

model_type = "large"
file_name  = "sample2"

RESULTS_SAVE_FILE = f"/Users/tsy/Documents/projects/py/tools/auto_memo_taker/text_transcripts/{file_name}/{file_name}_{model_type}.txt"
AUDIO_FILE = f"/Users/tsy/Documents/projects/py/tools/auto_memo_taker/mp3_source/{file_name}.m4a"


def write_to_file(file_path, data):
    with open(file_path, 'a') as file:
        file.write(data)


def touch_file(target_file):
    target_name = os.path.basename(target_file)
    if not os.path.isfile(target_file):
        _ = subprocess.run(["touch", target_file])
        print(f"{target_name} has been created!")
    else:
        print(f"{target_name} has already been existed.")


def audio2text():
    
    if not os.path.isfile(AUDIO_FILE):
        print("audio file does not exist!")
        return 
    
    print("testA")
    touch_file(RESULTS_SAVE_FILE)
    t0 = time.time()
    model = whisper.load_model(model_type)
    t1 = time.time()
    result = model.transcribe(audio=AUDIO_FILE)
    transcribed_text = result["text"]
    t2 = time.time()

    time_text = f"\nprocess of loading model uses {t1 - t0} sec." +\
                f"\nprocess of audio transcription uses {t2 - t1} sec." + "\n\n"
    write_to_file(RESULTS_SAVE_FILE, transcribed_text + time_text)



if __name__ == "__main__":

    try:
        for _ in range(2):
            audio2text()
    except Exception as e:
        pass