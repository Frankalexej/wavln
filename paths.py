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
word_seg_anno_path = bsc_path + "word_seg_anno/"

seq_seg_anno_path = bsc_path + "seq_seg_anno/"

phone_seg_random_log_path = bsc_path + "phone_seg_random_log/"
phone_seg_anno_log_path = bsc_path + "phone_seg_anno_log/"
word_seg_anno_log_path = bsc_path + "word_seg_anno_log/"
word_seg_anno_log_ref_path = bsc_path + "word_seg_anno_log_ref/"
word_plot_path = bsc_path + "word_plot_path/"
word_plot_res_path = bsc_path + "word_plot_res/"
word_plot_info_path = bsc_path + "word_plot_info/"  # for placing other infos related to each word
word_cluster_plot_path = bsc_path + "word_cluster_plot/"

model_save_dir = root_path + "model_save/"
model_eng_save_dir = model_save_dir + "eng/"
model_man_save_dir = model_save_dir + "man/"
# NOTE: don't put file paths here, only directory. 


# def run(): 
#     # list of all paths above
#     paths_list = [
#         bsc_path,
#         wav_path,
#         phones_path,
#         words_path,
#         test_path,
#         phones_extract_path,
#         words_extract_path,
#         stat_params_path,
#         segments_path,
#         phone_seg_anno_path,
#         phone_seg_random_path,
#         seq_seg_anno_path,
#         phone_seg_random_log_path,
#         phone_seg_anno_log_path, 
#         model_save_dir, 
#         model_eng_save_dir, 
#         model_man_save_dir, 
#         word_seg_anno_path, 
#         word_seg_anno_log_path
#     ]
#     for a_path in paths_list: 
#         os.makedirs(a_path, exist_ok = True)

def mk(dir): 
    os.makedirs(dir, exist_ok = True)


if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and not name.startswith("__"):
            globals()[name] = mk(value)
            # print(globals()[name])
