import pandas as pd
import numpy as np
import os
import librosa as rosa
from transformers import AutoProcessor, ClapModel
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torchaudio
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
import argparse
from muq import MuQ

corrupted_files = ["9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117", 
                   "8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233",
                   "6/norine_braun-now_and_zen-08-gently-117-146"]

def get_chunk_starts(total_samples, chunk_size):
    """Returns start indices for chunk_size-length windows covering the audio,
    with the last window anchored to the end (overlapping, not padded)."""
    if total_samples <= chunk_size:
        return [0]  # shorter than one chunk -- padding is unavoidable here, edge case
    starts = list(range(0, total_samples - chunk_size + 1, chunk_size))
    if starts[-1] + chunk_size < total_samples:
        starts.append(total_samples - chunk_size)  # end-anchored final window
    return starts

class HuggingFaceMuQ:
    def __init__(self, hf_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        muq_path = os.path.join(hf_path, "OpenMuQ", "MuQ-large-msd-iter")
        if os.path.exists(muq_path):
            self.model = MuQ.from_pretrained(muq_path).to(self.device)
        else:
            print("downloading and saving locally...")
            self.model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(self.device)
            self.model.save_pretrained(muq_path)

        self.model.eval()
        self.sample_rate = 24000
        self.max_batch_size = 2  # guard against OOM on very long tracks

    def get_audio_tokens(self, audio_path, chunk_secs=10):
        audio_data, sr = rosa.load(audio_path, sr=self.sample_rate)
        chunk_size = chunk_secs * self.sample_rate
        starts = get_chunk_starts(len(audio_data), chunk_size)
        chunks = [audio_data[s:s + chunk_size] for s in starts]  # all same length, no padding needed

        all_sequences = []

        for b in range(0, len(chunks), self.max_batch_size):
            batch = chunks[b:b + self.max_batch_size]

            # stack into (B, chunk_size) -- safe since every chunk is exactly chunk_size long
            wav_tensor = torch.tensor(np.stack(batch), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                output = self.model(wav_tensor, output_hidden_states=False)

            # last_hidden_state: (B, T_frames, dim) -- one forward pass, B chunks at once
            all_sequences.append(output.last_hidden_state.reshape(-1, output.last_hidden_state.shape[-1]).cpu())

        full_sequence = torch.cat(all_sequences, dim=0)  # (T_total, dim)
        print(f"MuQ full sequence for {os.path.basename(audio_path)}: {full_sequence.shape}")

        return full_sequence

class HuggingFaceMert:
    def __init__(self, hf_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mert_path = os.path.join(hf_path, "m-a-p", "MERT-v1-95M")

        if os.path.exists(mert_path):
            print("ja existe - loading local weights")
            self.model = AutoModel.from_pretrained(mert_path, trust_remote_code=True, local_files_only=True).to(self.device)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_path, trust_remote_code=True, local_files_only=True)
        else:
            print("downloading and saving locally...")
            self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(self.device)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

            # Fixed variable references
            self.model.save_pretrained(mert_path)
            self.processor.save_pretrained(mert_path)

        self.model.eval()
        self.max_batch_size = 2  # guard against OOM on very long tracks

    def get_audio_tokens(self, audio_path, chunk_secs=10):
        waveform, sr = torchaudio.load(audio_path)
        target_sr = self.processor.sampling_rate  # 24000
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        waveform = waveform.mean(dim=0)  # 1D, length = actual sample count

        chunk_size = chunk_secs * target_sr
        starts = get_chunk_starts(waveform.shape[0], chunk_size)
        chunks = [waveform[s:s + chunk_size] for s in starts]  # all same length, no padding needed

        all_sequences = []

        for b in range(0, len(chunks), self.max_batch_size):
            batch = chunks[b:b + self.max_batch_size]

            # processor accepts a list of 1D arrays/tensors and batches them
            inputs = self.processor(
                [c.numpy() for c in batch],
                sampling_rate=target_sr,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=False)

            # last_hidden_state: (B, T_frames, 768)
            all_sequences.append(outputs.last_hidden_state.reshape(-1, outputs.last_hidden_state.shape[-1]).cpu())

        full_sequence = torch.cat(all_sequences, dim=0)  # (T_total, 768)
        print(f"MERT full sequence for {os.path.basename(audio_path)}: {full_sequence.shape}")

        return full_sequence

class HuggingFaceClap:
    def __init__(self, hf_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(f"{hf_path}/laion/clap-htsat-unfused", local_files_only=True).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(f"{hf_path}/laion/clap-htsat-unfused", local_files_only=True)
        self.sample_rate = 48000
        self.max_batch_size = 10  # guard against OOM on very long tracks


    def get_audio_tokens(self, audio_path, chunk_secs=10):
        audio_data, sr = rosa.load(audio_path, sr=self.sample_rate)
        chunk_size = chunk_secs * self.sample_rate
        starts = get_chunk_starts(len(audio_data), chunk_size)
        chunks = [audio_data[s:s + chunk_size] for s in starts]

        chunk_sequences = []
        chunk_embeddings = []

        for b in range(0, len(chunks), self.max_batch_size):
            batch = chunks[b:b + self.max_batch_size]
            
            # The processor accepts a list of 1D numpy arrays.
            inputs = self.processor(
                text=[""] * len(batch),
                audios=batch,
                return_tensors="pt",
                padding=True,
                sampling_rate=self.sample_rate,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            audio_tokens = outputs.audio_model_output.last_hidden_state  # (1, 768, 2, 32)
            time_sequence = audio_tokens.mean(dim=2).permute(0, 2, 1)    # (1, 32, 768)

            # flatten batch+time into one sequence, preserving chunk order
            chunk_sequences.append(time_sequence.reshape(-1, time_sequence.shape[-1]).cpu())  # (B*32, 768)
            chunk_embeddings.append(outputs.audio_embeds.cpu())  # (B, 512)


        full_sequence = torch.cat(chunk_sequences, dim=0)  # (T_total, 768)
        global_emb = torch.cat(chunk_embeddings, dim=0).mean(dim=0, keepdim=True)  # (1, 512)

        print(f"CLAP full-song sequence for {os.path.basename(audio_path)}. Full_sequence shape: {full_sequence.shape}, Global embedding shape: {global_emb.shape}")

        return full_sequence


def clap_worker_fn(ModelClass, proc_id, audio_files, audios_dir, features_dir, hf_path):
    """Worker process function: Instantiates its own independent model on the single GPU
    and processes its allocated slice of audio files.
    """


    model = ModelClass(hf_path=hf_path)
    total_files = len(audio_files)
    print(f"[Process {proc_id}] Assigned {total_files} audio files.")

    #----------------------Extract features from audios ----------------#
    for i, file in enumerate(audio_files):
        file_name = os.path.splitext(file)[0] + ".npy"
        save_path = os.path.join(features_dir, file_name)

        tokens = model.get_audio_tokens(os.path.join(audios_dir, file))
        np.save(save_path, tokens.squeeze().detach().cpu().numpy())

        print(f"[Process {proc_id}] Progress: {i + 1}/{total_files}")

def worker_fn(ModelClass, proc_id, audio_files, audios_dir, features_dir, hf_path):
    """Worker process function: Instantiates its own independent model on the single GPU
    and processes its allocated slice of audio files.
    """
    model = ModelClass(hf_path=hf_path)
    total_files = len(audio_files)
    print(f"[Process {proc_id}] Assigned {total_files} audio files.")

    #----------------------Extract features from audios ----------------#
    for i, file in enumerate(audio_files):
        file_name = os.path.splitext(file)[0] + ".npy"
        save_path = os.path.join(features_dir, file_name)

        emb = model.get_audio_tokens(os.path.join(audios_dir, file))
        np.save(save_path, emb.squeeze().detach().cpu().numpy())   
        print(f"[Process {proc_id}] Progress: {i + 1}/{total_files}")

def main():
    #----------------------------- Argument Parsing ---------------------------------- #
    parser = argparse.ArgumentParser(description="Extract embeddings from audio files.")
    parser.add_argument("--model", type=str, required=True, help="Model name: 'mert', 'clap' or 'muq'")
    args = parser.parse_args()

    features_folder = f'{args.model}_features'
    if args.model == 'mert':
        ModelClass = HuggingFaceMert
        worker_func = worker_fn
    elif args.model == 'clap':
        ModelClass = HuggingFaceClap
        worker_func = clap_worker_fn  # Use the specialized worker for CLAP
    elif args.model == 'muq':
       ModelClass = HuggingFaceMuQ
       worker_func = worker_fn
    else:
        raise ValueError("Invalid model name. Choose from 'mert', 'clap', or 'muq'.")

    # ----------------------------- Environment & Config ---------------------------------- #
    load_dotenv()
    DATASETS_FEATURES_FOLDER = os.environ["DATASETS_FEATURES_FOLDER"]
    datasets = ['jamendo', 'magnatagatune']
    HF_PATH = os.environ["HF_PATH"]
    NUM_PROCESSES = 10

    #main loop
    for dataset in datasets:
        print(f"\nProcessing: {dataset}")

        root_dir = os.path.join(DATASETS_FEATURES_FOLDER, dataset)
        audios_dir = os.path.join(root_dir, "audios")

        features_dir = os.path.join(root_dir, features_folder)
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
        print(f"Found {total_audios} pending files for {dataset}")

        if total_audios == 0:
            continue

        # Split files evenly across processes using strided slicing
        chunks = [audio_files[i::NUM_PROCESSES] for i in range(NUM_PROCESSES)]

        # Launch processes
        processes = []
        for proc_id in range(NUM_PROCESSES):
            p = mp.Process(
                target=worker_func,
                args=(
                    ModelClass,
                    proc_id,
                    chunks[proc_id],
                    audios_dir,
                    features_dir,
                    HF_PATH,
                ),
            )
            p.start()
            processes.append(p)

        # Join processes to wait for the current dataset to complete
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()

