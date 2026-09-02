import pandas as pd
import numpy as np
import os
import torch.multiprocessing as mp
from dotenv import load_dotenv
from essentia.standard import *
import essentia
import numpy

def load_audio(filename, sampleRate=12000, segment_duration=None):
    audio = MonoLoader(filename=filename, sampleRate=sampleRate, resampleQuality=4)()

    if segment_duration:
        segment_duration = round(segment_duration*sampleRate)
        segment_start = (len(audio) - segment_duration) // 2
        segment_end = segment_start + segment_duration
    else:
        segment_start = 0
        segment_end = len(audio)

    if segment_start < 0 or segment_end > len(audio):
        raise ValueError('Segment duration is larger than the input audio duration')

    return audio[segment_start:segment_end]


def melspectrogram(audio,
                   sampleRate=12000, frameSize=512, hopSize=256,
                   window='hann', zeroPadding=0, center=True,
                   numberBands=96, lowFrequencyBound=0, highFrequencyBound=None,
                   weighting='linear', warpingFormula='slaneyMel',
                   normalize='unit_tri'):

    if highFrequencyBound is None:
        highFrequencyBound = sampleRate/2

    windowing = Windowing(type=window, normalized=False, zeroPadding=zeroPadding)
    spectrum = Spectrum()
    melbands = MelBands(numberBands=numberBands,
                        sampleRate=sampleRate,
                        lowFrequencyBound=lowFrequencyBound,
                        highFrequencyBound=highFrequencyBound,
                        inputSize=(frameSize+zeroPadding)//2+1,
                        weighting=weighting,
                        normalize=normalize,
                        warpingFormula=warpingFormula,
                        type='power')
    amp2db = UnaryOperator(type='lin2db', scale=2)

    pool = essentia.Pool()
    for frame in FrameGenerator(audio,
                                frameSize=frameSize, hopSize=hopSize,
                                startFromZero=not center):
        pool.add('mel', amp2db(melbands(spectrum(windowing(frame)))))

    return pool['mel'].T


def analyze(audio_file, npy_file, full_audio=True):
    if full_audio:
      # Analyze full audio duration.
      segment_duration=None
    else:
      # Duration for the Choi's VGG model.
      segment_duration=29.1

    audio = load_audio(audio_file, segment_duration=segment_duration)
    mel = melspectrogram(audio)
    numpy.save(npy_file, mel, allow_pickle=False)
    return

corrupted_files = ["9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117", 
                   "8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233",
                   "6/norine_braun-now_and_zen-08-gently-117-146"]

def worker_fn(proc_id, audio_files, audios_dir, features_dir):
    """Worker process function: Processes its allocated slice of audio files

    and extracts Mel-spectrograms.
    """
    total_files = len(audio_files)
    print(f"[Process {proc_id}] Assigned {total_files} audio files.")

    for i, file in enumerate(audio_files):
        file_name = os.path.splitext(file)[0] + ".npy"
        save_path = os.path.join(features_dir, file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        audio_path = os.path.join(audios_dir, file)

        try:
            analyze(audio_path, save_path, full_audio=False)
            print(f"[Process {proc_id}] Progress: {i + 1}/{total_files}")
        except Exception as e:
            print(f"[Process {proc_id}] Error processing {file}: {e}")

def main():
    #extract melspecs from the magna tag a tune dataset

    load_dotenv()
    DATASETS_FEATURES_FOLDER = os.environ["DATASETS_FEATURES_FOLDER"]
    HF_PATH = os.environ["HF_PATH"]
    NUM_PROCESSES = 10

    #main loop
    root_dir = os.path.join(DATASETS_FEATURES_FOLDER, 'magnatagatune')
    audios_dir = os.path.join(root_dir, "audios")

    features_dir = os.path.join(root_dir, "melspecs")
    os.makedirs(features_dir, exist_ok=True)

    #Create subdirs for extracted features and create a list of audio files to process
    subfolders = os.listdir(audios_dir)
    audio_files = []
    for subfolder in subfolders:
        subfolder_path = os.path.join(audios_dir, subfolder)

        #create save subdir
        os.makedirs(os.path.join(features_dir, subfolder), exist_ok=True)

        #create a list of audio files to process - only unprocessed files
        for f in os.listdir(subfolder_path):
            rel_file_path = os.path.join(subfolder, f)
            save_file_name = os.path.splitext(rel_file_path)[0] + ".npy"

            #if file is corrupted, skip it
            if os.path.splitext(rel_file_path)[0] in corrupted_files:
                print(f"Skipping corrupted file: {rel_file_path}")
                continue

            save_path = os.path.join(features_dir, save_file_name)
            if not os.path.exists(save_path):
                audio_files.append(rel_file_path)

    #----------------------Run the worker function in parallel--------------------------#
    total_audios = len(audio_files)
    if total_audios == 0:
        print("No audio files to process. Exiting.")
        return

    # Split files evenly across processes using strided slicing
    chunks = [audio_files[i::NUM_PROCESSES] for i in range(NUM_PROCESSES)]

    processes = []
    for proc_id in range(NUM_PROCESSES):
        p = mp.Process(
            target=worker_fn,
            args=(
                proc_id,
                chunks[proc_id],
                audios_dir,
                features_dir,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()

