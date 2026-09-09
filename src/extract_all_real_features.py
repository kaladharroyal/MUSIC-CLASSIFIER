"""
Batch Real Feature Extractor for MTG-Jamendo & MagnaTagATune
Iterates through local audio folders, extracts 23 signal features + 7 mid-level features,
and outputs authentic feature CSVs to Data/processed/.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import librosa

from essentia_features import extract_essentia_signal_features
from midlevel_vggish import MidLevelVGGish, get_midlevel_feature_names

def process_dataset_audio(audio_root, output_csv, max_tracks=500):
    """
    Scans audio_root for .mp3 / .wav / .ogg / .flac files, extracts features,
    and saves to output_csv.
    """
    print(f"\nScanning audio files in '{audio_root}'...")
    audio_extensions = ('*.mp3', '*.wav', '*.ogg', '*.flac', '*.au')
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_root, "**", ext), recursive=True))
        
    print(f"Found {len(audio_files)} total audio tracks.")
    if len(audio_files) == 0:
        print(f"Warning: No audio files found under '{audio_root}'")
        return None

    if max_tracks and len(audio_files) > max_tracks:
        print(f"Subsampling to {max_tracks} tracks for fast, efficient feature extraction...")
        random.seed(42)
        audio_files = random.sample(audio_files, max_tracks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midlevel_model = MidLevelVGGish().to(device)
    midlevel_model.eval()

    records = []
    mid_names = get_midlevel_feature_names()

    for idx, fpath in enumerate(audio_files):
        fname = os.path.basename(fpath)
        if (idx + 1) % 25 == 0 or (idx + 1) == len(audio_files):
            print(f"[{idx+1}/{len(audio_files)}] Processing {fname}...")

        try:
            # 1. Signal features (23-dim)
            sig_feats = extract_essentia_signal_features(fpath)
            
            # 2. Mid-level features (7-dim) via spectrogram + MidLevelVGGish
            y, sr = librosa.load(fpath, sr=22050, duration=30.0)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=2048, hop_length=1024)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Pad / Crop time dimension to 649 frames
            if mel_spec_db.shape[1] < 649:
                pad_width = 649 - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :649]
                
            tensor_spec = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                mid_preds = midlevel_model(tensor_spec).cpu().numpy().squeeze()
                
            for m_name, val in zip(mid_names, mid_preds):
                sig_feats[m_name] = float(val)
                
            sig_feats['filename'] = fname
            sig_feats['filepath'] = fpath
            records.append(sig_feats)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    df_out = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(df_out)} extracted real feature vectors to '{output_csv}'!")
    return df_out

def main():
    jamendo_audio_root = os.path.join("Data", "research_external", "MTG-Jamendo", "mtg-jamendo_gtzan_subset")
    jamendo_csv_out = os.path.join("Data", "processed", "jamendo_real_features.csv")
    process_dataset_audio(jamendo_audio_root, jamendo_csv_out, max_tracks=300)

    mtat_audio_root = os.path.join("Data", "research_external", "MagnaTagATune", "MagnaTagATune")
    mtat_csv_out = os.path.join("Data", "processed", "mtat_real_features.csv")
    process_dataset_audio(mtat_audio_root, mtat_csv_out, max_tracks=300)

if __name__ == "__main__":
    main()
