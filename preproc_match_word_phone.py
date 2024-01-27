import pandas as pd
import os

from paths import *
from misc_tools import PathUtils as PU
import argparse


# def match_word_phone(words_data, phones_data):
#     # Convert words and phones data to DataFrame if they are not already
#     df_words = pd.DataFrame(words_data) if not isinstance(words_data, pd.DataFrame) else words_data
#     df_phones = pd.DataFrame(phones_data) if not isinstance(phones_data, pd.DataFrame) else phones_data

#     # Add new columns to the phones DataFrame
#     df_phones['word_id'] = None
#     df_phones['word'] = None
#     df_phones['in_id'] = None

#     # Iterate over each phone and find matching word
#     for idx, phone in df_phones.iterrows():
#         matching_words = df_words[(df_words['startTime'] <= phone['startTime']) & (df_words['endTime'] >= phone['endTime'])]
        
#         if not matching_words.empty:
#             word = matching_words.iloc[0]  # Get the first matching word
#             df_phones.at[idx, 'word_id'] = word['id']
#             df_phones.at[idx, 'word'] = word['segment']
#             # Counting the internal index (in_id) of the phone within the word
#             df_phones.at[idx, 'in_id'] = (df_phones[(df_phones['word_id'] == word['id']) & (df_phones.index <= idx)]).shape[0]

#     return df_phones

def match_word_phone(df_words, df_phones):
    # input must be dataframe

    # Add new columns to the phones DataFrame
    df_phones['word_id'] = None
    df_phones['word'] = None
    df_phones['in_id'] = None

    # Iterate over each phone and find matching word within the same audio file
    for idx, phone in df_phones.iterrows():
        matching_words = df_words[(df_words['file'] == phone['file']) &
                                  (df_words['startTime'] <= phone['startTime']) & 
                                  (df_words['endTime'] >= phone['endTime'])]
        
        if not matching_words.empty:
            word = matching_words.iloc[0]  # Get the first matching word
            df_phones.at[idx, 'word_id'] = word['id']
            df_phones.at[idx, 'word'] = word['segment']
            # Counting the internal index (in_id) of the phone within the word
            df_phones.at[idx, 'in_id'] = sum((df_phones['word_id'] == word['id']) & (df_phones.index <= idx))

    return df_phones

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--guide_path', '-gp', type=str, default="..//src/eng/", help="Path that holds the guide files")
    parser.add_argument('--word_guide_name', '-wgn', type=str, default="word_guide.csv", help="Name of word guide file")
    parser.add_argument('--phone_guide_name', '-pgn', type=str, default="phone_guide.csv", help="Name of phone guide file")
    parser.add_argument('--matched_phone_guide_name', '-mpgn', type=str, default="matched_phone_guide.csv", help="Name of matched output guide file")
    args = parser.parse_args()
    # Load the words and phones data
    df_words = pd.read_csv(os.path.join(args.guide_path, args.word_guide_name))
    df_phones = pd.read_csv(os.path.join(args.guide_path, args.phone_guide_name))

    # Match the phones with the words
    df_phones = match_word_phone(df_words, df_phones)

    # Save the new phones data
    df_phones.to_csv(os.path.join(args.guide_path, args.matched_phone_guide_name), index=False)