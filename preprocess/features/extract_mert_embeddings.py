import argparse
import numpy as np
import librosa
import pretty_midi
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

def get_audio_dlnco(audio_path, fs=22050, feature_rate=22050/64):
    pass

def get_midi_dlnco(midi_path, duration_sec=None, feature_rate=22050/64):
    pass
    
    

def process_audio(audio, audio_dir, audio_dlnco_dir):
    """Worker function for a single audio file."""
    basename = os.path.splitext(audio)[0]
    save_path = os.path.join(audio_dlnco_dir, basename + ".npy")

    if not os.path.exists(save_path):
        audio_path = os.path.join(audio_dir, audio)
        dlnco = get_audio_dlnco(audio_path)
        np.save(save_path, dlnco)

    return audio

def process_midi(midi, corrupt_midi_dir, corrupt_midi_dlnco_dir):
    """Worker function for a single MIDI file."""
    basename = os.path.splitext(midi)[0]
    dlnco_save_path = os.path.join(corrupt_midi_dlnco_dir, basename + ".npy")

    if not os.path.exists(dlnco_save_path):
        midi_path = os.path.join(corrupt_midi_dir, midi)
        dlnco = get_midi_dlnco(midi_path)
        np.save(dlnco_save_path, dlnco)

    return midi

def run_parallel(files, worker_function, worker_args, max_workers, description, chunksize=1):
    total = len(files)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            worker_function,
            files,
            *[
                [arg] * total
                for arg in worker_args
            ],
            chunksize=chunksize,
        )

        for i, result in enumerate(results, start=1):
            print(f"{description}: "f"{i}/{total} | {result}")

def main():
    # ----------------------------- Argparse Setup ---------------------------------- #
    parser = argparse.ArgumentParser(description="Batch Audio Transcription Pipeline")
    parser.add_argument(
        "--original",
        action="store_true",
        help="Extract only from the original data."
    )
    args = parser.parse_args()
    # ----------------------------- Environment & Config ---------------------------------- #
    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    datasets = ['magnatagatune', 'jamendo']

    #----------------------------- Process Features ---------------------------------- #

    cpu_count = os.cpu_count() or 1
    max_workers = max(1, int(cpu_count * 0.8))
    print(f"Running on {max_workers} CPU cores.")

    for dataset in datasets:
        print(f"\nProcessing: {dataset}")

        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        original_audio_dir = os.path.join(root_dir, "audios")
        original_audio_dlnco_dir = os.path.join(root_dir, "audios_dlnco")

        if args.original:
            corrupt_midi_dir = os.path.join(root_dir, "midis")
            corrupt_midi_dlnco_dir = os.path.join(root_dir, "original_midi_dlnco")
        else:
            corrupt_midi_dir = os.path.join(root_dir, "corrupt_midi")
            corrupt_midi_dlnco_dir = os.path.join(root_dir, "corrupt_midi_dlnco")

        os.makedirs(corrupt_midi_dlnco_dir, exist_ok=True)
        os.makedirs(original_audio_dlnco_dir, exist_ok=True)
        

        # -------------------------
        # Audio DLNCO
        # -------------------------

        audio_files = [f for f in os.listdir(original_audio_dir) if f.endswith(".wav") or f.endswith(".mp3")]

        run_parallel(
            files=audio_files,
            worker_function=process_audio,
            worker_args=(
                original_audio_dir,
                original_audio_dlnco_dir,
            ),
            max_workers=max_workers,
            description="Audio DLNCO",
            chunksize=1,
        )

        # -------------------------
        # MIDI DLNCO
        # -------------------------

        midi_files = [f for f in os.listdir(corrupt_midi_dir) if f.endswith(".mid") or f.endswith(".midi")]

        run_parallel(
            files=midi_files,
            worker_function=process_midi,
            worker_args=(
                corrupt_midi_dir,
                corrupt_midi_dlnco_dir,
            ),
            max_workers=max_workers,
            description="MIDI DLNCO",
            chunksize=10,
        )


if __name__ == "__main__":
    main()