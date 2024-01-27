"""
Other modifications on generated guide
"""
import pandas as pd
import os
import numpy as np

from paths import *
from ssd_paths import *
from misc_tools import PathUtils as PU
from misc_tools import AudioCut
from misc_progress_bar import draw_progress_bar


def modify_guide(src_path, target_dir, target_prefix): 
    assert (PU.path_exist(src_path))

    total_df = pd.read_csv(src_path)

    speakers = total_df["file"].str.split("-").str[0].unique().tolist()

    np.random.shuffle(speakers)
    num_speakers = len(speakers)
    ratios = [0.8, 0.1, 0.1]
    train_speakers = speakers[:int(num_speakers*ratios[0])]
    val_speakers = speakers[int(num_speakers*ratios[0]):int(num_speakers*(ratios[0] + ratios[1]))]
    test_speakers = speakers[int(num_speakers*(ratios[0] + ratios[1])):]
    
    # create output tensors for each set of speakers
    train_df = total_df[total_df["file"].str.split("-").str[0].isin(train_speakers)]
    val_df = total_df[total_df["file"].str.split("-").str[0].isin(val_speakers)]
    test_df = total_df[total_df["file"].str.split("-").str[0].isin(test_speakers)]

    train_df.to_csv(os.path.join(target_dir, target_prefix + "train.csv"), index=False)
    val_df.to_csv(os.path.join(target_dir, target_prefix + "validation.csv"), index=False)
    test_df.to_csv(os.path.join(target_dir, target_prefix + "test.csv"), index=False)
    return



if __name__ == "__main__": 
    # modify_guide(os.path.join(src_, "guide.csv"), os.path.join(src_, "guide_mod.csv"))
    modify_guide(os.path.join(ssrc_, "guide_mod_pathed.csv"), suse_, "guide_")