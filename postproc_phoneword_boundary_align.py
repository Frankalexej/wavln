import pandas as pd
import os
from paths import word_seg_anno_log_path, phone_seg_anno_log_path, compound_word_log_path, bsc_path
from misc_progress_bar import draw_progress_bar

# Now I want to make it file by file, not the whole one which is bad
def align(word_log_path, phone_dir, compound_dir): 
    # Step 1: Read wordlog.csv and phonelog.csv into DataFrames
    wordlog_df = pd.read_csv(word_log_path)
    wordlog_df['produced_segments_clean'] = wordlog_df['produced_segments'].str.strip()
    wordlog_df = wordlog_df[(wordlog_df["produced_segments_clean"]!="") & (wordlog_df["produced_segments_clean"].notna())]

    current_phonelog_df = None  # will be renewed along the way
    current_rec = ""  # init

    # Step 2-6: Iterate through wordlog_df and perform the checks
    results = []
    total_length = len(wordlog_df)

    for idx, word_row in wordlog_df.iterrows():
        rec = word_row['rec']
        if rec != current_rec: 
            current_phonelog_df = pd.read_csv(os.path.join(phone_dir, rec + ".csv"))

        start_time_word = word_row['start_time']
        end_time_word = word_row['end_time']
        produced_segments = word_row['produced_segments_clean'].split()

        # Step 4: Filter phonelog_df for matching 'rec' and time range
        # (phonelog_df['rec'] == rec) & 
        phonelog_filtered = current_phonelog_df[(current_phonelog_df['start_time'] >= start_time_word) & (current_phonelog_df['end_time'] <= end_time_word)]

        # Step 5: Check if phonemes are included in the produced segments and follow the same order
        phonemes = phonelog_filtered['token'].tolist()
        phoneme_endtimes = phonelog_filtered['end_time'].tolist()
        if phonemes == produced_segments:
            match_status = 1
        else:
            match_status = 0

        # Step 6: Store the results
        try: 
            results.append({
                'rec': rec,
                'idx': word_row['idx'], 
                'start_time': start_time_word,
                'end_time': end_time_word,
                'token': word_row['token'], 
                'duration': word_row['duration'], 
                'n_frames': word_row['n_frames'], 
                'produced_segments': ' '.join(produced_segments),
                'phonemes': ' '.join(phonemes),
                'phoneme_endtimes': ' '.join(map(str, phoneme_endtimes)),
                'match_status': match_status
            })
        except Exception: 
            results.append({
                'rec': rec,
                'idx': word_row['idx'], 
                'start_time': start_time_word,
                'end_time': end_time_word,
                'token': word_row['token'], 
                'duration': word_row['duration'], 
                'n_frames': word_row['n_frames'], 
                'produced_segments': ' '.join(produced_segments),
                'phonemes': '',
                'phoneme_endtimes': ' '.join(map(str, phoneme_endtimes)),
                'match_status': 2
            })
        if idx % 10 == 0: 
            draw_progress_bar(idx, total_length)

    # Create a DataFrame from the results
    result_df = pd.DataFrame(results)

    # Save the results to a CSV file
    result_df.to_csv(os.path.join(compound_dir, "log.csv"), index=False)


if __name__ == "__main__": 
    align(word_log_path=os.path.join(word_seg_anno_log_path, "log.csv"), 
            phone_dir=phone_seg_anno_log_path, 
            compound_dir=compound_word_log_path)