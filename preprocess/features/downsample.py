import math
import multiprocessing as mp
import os
import numpy as np
import torch
from dotenv import load_dotenv
import torch.nn.functional as F

def downsample_sequence(seq, target_len=128):
    T, dim = seq.shape
    if T <= target_len:
        return seq
    x = seq.T.unsqueeze(0)                      # (1, dim, T)
    pooled = F.adaptive_avg_pool1d(x, target_len)  # (1, dim, target_len)
    return pooled.squeeze(0).T                   # (target_len, dim)


def worker_func(proc_id, file_chunk, features_dir, downsampled_dir, max_len=128):
    """Worker function to load, downsample, and save feature files in parallel."""
    for idx, rel_file_path in enumerate(file_chunk):
        in_path = os.path.join(features_dir, rel_file_path)

        # Output path handling (.npy format)
        save_file_name = os.path.splitext(rel_file_path)[0] + ".npy"
        out_path = os.path.join(downsampled_dir, save_file_name)

        #load the entire np array
        arr = np.load(in_path)
        seq = torch.from_numpy(arr).float()

        #downsample it   
        downsampled_seq = downsample_sequence(seq, target_len=max_len)

        # Save downsampled numpy array
        np.save(out_path, downsampled_seq.half().numpy())

        print(f"[Proc {proc_id}] Progress: {idx + 1}/{len(file_chunk)} files completed")


def main():
    # ----------------------------- Environment & Config ---------------------------------- #
    load_dotenv()
    DATASETS_FEATURES_FOLDER = os.environ["DATASETS_FEATURES_FOLDER"]
    datasets = ["jamendo", "magnatagatune"]
    models = ["clap", "mert", "muq"]
    NUM_PROCESSES = 4

    # Main loop
    for dataset in datasets:
        print(f"\nProcessing: {dataset}")
        root_dir = os.path.join(DATASETS_FEATURES_FOLDER, dataset)

        for model in models:
            features_dir = os.path.join(root_dir, f"{model}_features")
            downsampled_dir = os.path.join(root_dir, f"downsampled_{model}_features")
            os.makedirs(downsampled_dir, exist_ok=True)

            # Gather pending files
            features_files = []
            for subfolder in os.listdir(features_dir):
                subfolder_path = os.path.join(features_dir, subfolder)

                os.makedirs(os.path.join(downsampled_dir, subfolder), exist_ok=True)

                for f in os.listdir(subfolder_path):
                    rel_file_path = os.path.join(subfolder, f)
                    save_file_name = (os.path.splitext(rel_file_path)[0] + ".npy")
                    save_path = os.path.join(downsampled_dir, save_file_name)

                    if not os.path.exists(save_path):
                        features_files.append(rel_file_path)

            # ---------------------- Run worker processes -------------------------- #
            total_audios = len(features_files)
            print(f"Found {total_audios} pending files for {dataset} ({model})")

            if total_audios == 0:
                continue

            # Split files evenly across processes using strided slicing
            chunks = [features_files[i::NUM_PROCESSES] for i in range(NUM_PROCESSES)]

            # Launch processes
            processes = []
            for proc_id in range(NUM_PROCESSES):
                if not chunks[proc_id]:
                    continue

                p = mp.Process(
                    target=worker_func,
                    args=(
                        proc_id,
                        chunks[proc_id],
                        features_dir,
                        downsampled_dir,
                    ),
                )
                p.start()
                processes.append(p)

            # Join processes to wait for completion before moving to next model/dataset
            for p in processes:
                p.join()


if __name__ == "__main__":
    print("Starting parallel downsampling...")
    main()