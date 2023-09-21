### This file is designed to check any empty cut. 
### Because there are occasionally some cuts that are but the length is strangely normal. 
import pandas as pd
import torchaudio
import os
from paths import phone_seg_anno_path, phone_seg_anno_log_path
from misc_progress_bar import draw_progress_bar

def open_and_log(rec_dir, log_dir): 
    # run through all recordings and log down n_frames
    control_file = pd.read_csv(log_dir)

    # Extract the "rec" and "idx" columns
    rec_col = control_file['rec'].astype(str)
    idx_col = control_file['idx'].astype(str).str.zfill(8)

    merged_col = rec_col + '_' + idx_col + ".wav"
    merged_list = merged_col.tolist()
    total_length = len(merged_list)

    n_frames_list = []

    for idx, audio_name in enumerate(merged_list): 
        rec, sample_rate = torchaudio.load(os.path.join(rec_dir, audio_name))
        n_frames_list.append(rec.size(1))

        if idx % 50 == 0: 
            draw_progress_bar(idx, total_length)
    

    control_file["n_frames"] = n_frames_list
    control_file.to_csv(log_dir, index=False)


if __name__ == "__main__": 
    open_and_log(phone_seg_anno_path, os.path.join(phone_seg_anno_log_path, "log.csv"))