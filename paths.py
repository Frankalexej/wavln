# This file contains path configurations for a speech recognition project.
import os


# root_path = "A:/Program Data/Python Projects/sequence_learning/"
root_path = "../"
bsc_path = root_path + "src/bsc/"
wav_path = bsc_path + "wav/"
phones_path = bsc_path + "phones/"
words_path = bsc_path + "words/"
test_path = root_path + "src/test/"

phones_extract_path = bsc_path + "phones_extract/"
words_extract_path = bsc_path + "words_extract/"
stat_params_path = bsc_path + "stat_params/"
segments_path = bsc_path + "segments/"


phone_seg_anno_path = bsc_path + "phone_seg_anno/"
phone_seg_random_path = bsc_path + "phone_seg_random/"
seq_seg_anno_path = bsc_path + "seq_seg_anno/"

phone_seg_random_log_path = bsc_path + "phone_seg_random_log/"
phone_seg_anno_log_path = bsc_path + "phone_seg_anno_log/"

model_save_dir = root_path + "model_save/"
model_eng_save_dir = model_save_dir + "eng/"
model_man_save_dir = model_save_dir + "man/"


def run(): 
    # list of all paths above
    paths_list = [
        bsc_path,
        wav_path,
        phones_path,
        words_path,
        test_path,
        phones_extract_path,
        words_extract_path,
        stat_params_path,
        segments_path,
        phone_seg_anno_path,
        phone_seg_random_path,
        seq_seg_anno_path,
        phone_seg_random_log_path,
        phone_seg_anno_log_path
    ]
    for a_path in paths_list: 
        os.makedirs(a_path, exist_ok = True)

if __name__ == "__main__": 
    run()
