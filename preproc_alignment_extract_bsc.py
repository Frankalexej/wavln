'''
# Alignment Extraction (English)

This Jupyter Notebook is designed to extract annotations from alignment files (in either .phones or .words format) and prepare for sound cutting. Specifically, the notebook will:

- Open .phones or .words alignment files
- Extract the annotations from these files
- Write the entries into a Pandas dataframe
- Save the dataframe as an Excel file
- Define classes and functions to prepare for sound cutting

! Only for bsc at the moment
'''

import pandas as pd
import os
from paths import *  # all paths
import sys




"""Functions"""
def remove_semicolon(s):
    """
    Removes the trailing semicolon from a given string.
    If the string does not end with a semicolon, it is returned unchanged.

    Args:
        my_string: A string to be processed.

    Returns:
        The same string with the trailing semicolon removed (if there was one).
    """
    if s.endswith(";"):
        s = s[:-1]
    return s

def line_starts_with_semicolon(line):
    """Determines whether a line starts with `;`.

    Args:
        line (str): A string representing a line of text.

    Returns:
        bool: True if the line starts with `;`, False otherwise.
    """
    return line.strip().startswith(';')

def errorless_get(arr, index, default=""): 
    """Applies indexing that avoids error
    """
    if len(arr) <= index: 
        return default
    else: 
        return arr[index]

def extract(path, from_words=False):
    """
    Extracts end times and tokens from a file.

    Args:
        path: The path to the input file.
        from_words: if extracting from .words files. 

    Returns:
        A tuple containing a list of end times and a list of corresponding tokens.
    """
    f = open(path)
    lines = f.readlines()
    end_times = []
    tokens = []
    theory_segs = []
    produce_segs = []

    putin = False
    for line in lines:
        if putin:
            if line_starts_with_semicolon(line): 
                continue
            splitted = line.split() 
            if len(splitted) == 0: 
                continue
            elif len(splitted) < 3: 
                end_times.append(float(splitted[0]))
                tokens.append("")
                if from_words: 
                    # if extracting from .words files, 
                    # notify that there is no assigned words and therefore no such segments. 
                    theory_segs.append("")
                    produce_segs.append("")
            else: 
                end_times.append(float(splitted[0]))
                tokens.append(remove_semicolon(splitted[2]))
                if from_words: 
                    semicolon_splitted = line.split(";")
                    theory_segs.append(errorless_get(semicolon_splitted, 1))
                    produce_segs.append(errorless_get(semicolon_splitted, 2))
                if splitted[2] == "{E_TRANS}": 
                    break   # time to stop
                    
        if "#" in line:
            putin = True

    f.close()
    if from_words: 
        return end_times, tokens, theory_segs, produce_segs
    return end_times, tokens, None, None

def create_dataframe(end_times, tokens, theory_segs=None, produce_segs=None):
    """
    Creates a pandas dataframe from lists of token end times and tokens.
    Calculates start times and durations for each token and adds these to the dataframe.

    Args:
        end_times (list): A list of token end times in seconds.
        tokens (list): A list of tokens.

    Returns:
        pandas.DataFrame: A dataframe with columns for start time, end time, token, and duration.
    """
    # Calculate start times
    start_times = [0.0] + end_times[:-1]
    
    # Calculate durations
    durations = [e - s for s, e in zip(start_times, end_times)]
    
    # Create dataframe
    df = pd.DataFrame({
        'start_time': start_times,
        'end_time': end_times,
        'token': tokens,
        'duration': durations
    })

    if theory_segs: 
        df["theory_segments"] = theory_segs
    if produce_segs: 
        df["produced_segments"] = produce_segs
    
    return df

def extract_and_create_dataframe(input_path, output_path, from_words=False):
    """
    Extracts token information from all .phones or .words files in a given input path,
    creates a pandas dataframe for each file, and outputs each dataframe to the corresponding
    filename in a given output path.

    Args:
        input_path (str): The path to the directory containing the .phones or .words files.
        output_path (str): The path to the directory where the resulting dataframes will be saved.
    """
    # Loop through all files in input path
    for file_name in os.listdir(input_path):
        if file_name.endswith('.phones') or file_name.endswith('.words'):
            # Extract token information
            end_times, tokens, theory_segs, produce_segs = extract(os.path.join(input_path, file_name), from_words=from_words)

            # Create dataframe
            df = create_dataframe(end_times, tokens, theory_segs=theory_segs, produce_segs=produce_segs)

            # Output dataframe to file in output path
            output_file_name = os.path.splitext(file_name)[0] + '.csv'
            df.to_csv(os.path.join(output_path, output_file_name), index=False)

def csv_bind(log_dir): 
    # List all the CSV files in the directory that start with 's'
    directory = log_dir
    csv_files = sorted([f for f in os.listdir(directory) if f.startswith('s') and f.endswith('.csv')])

    # Read and concatenate the CSV files using pandas
    dfs = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory, file))
        dfs.append(df)

    concatenated_df = pd.concat(dfs, ignore_index=True)

    # Save the concatenated dataframe as "log.csv"
    concatenated_df.to_csv(os.path.join(directory, 'log.csv'), index=False)

def run(run_type="", sub_type=""): 
    if run_type == "phone": 
        extract_and_create_dataframe(phones_path, phones_extract_path)
    elif run_type == "word": 
        extract_and_create_dataframe(words_path, words_extract_path, from_words=True)
    elif run_type == "bind": 
        try: 
            if sub_type == "phone": 
                csv_bind(phones_extract_path)
            elif sub_type == "word": 
                csv_bind(words_extract_path)
        except Exception: 
            print("Unsucessful! ")





if __name__ == "__main__": 
    if len(sys.argv) == 2: 
        run(sys.argv[1])
    elif len(sys.argv) == 3: 
        run(sys.argv[1], sys.argv[2])