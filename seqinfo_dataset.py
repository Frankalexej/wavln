import pandas as pd
from torch.utils.data import Dataset
import torch
import torchaudio
import os
from torch.nn.utils.rnn import pad_sequence
from torch import nn

# class PhoneDataset(Dataset):
#     """
#     A PyTorch dataset that loads cutted wave files from disk and returns input-output pairs for
#     training autoencoder. 
    
#     Version 3: wav -> mel
#     """
    
#     def __init__(self, load_dir, load_control_path, transform=None):
#         """
#         Initializes the class by reading a CSV file and merging the "rec" and "idx" columns.

#         The function reads the CSV file from the provided control path, extracts the "rec" and "idx" columns,
#         and concatenates the values from these columns using an underscore. It then appends the ".wav" extension
#         to each of the merged strings and converts the merged pandas Series to a list, which is assigned to
#         the 'dataset' attribute of the class.

#         Args:
#         load_dir (str): The directory containing the files to load.
#         load_control_path (str): The path to the CSV file containing the "rec" and "idx" columns.

#         Attributes:
#         dataset (list): A list of merged strings from the "rec" and "idx" columns, with the ".wav" extension.
#         """
#         control_file = pd.read_csv(load_control_path)
#         control_file = control_file[control_file['n_frames'] > 400]
#         control_file = control_file[control_file['duration'] <= 2.0]
        
#         # Extract the "rec" and "idx" columns
#         rec_col = control_file['rec'].astype(str)
#         idx_col = control_file['idx'].astype(str).str.zfill(8)
        
#         # Merge the two columns by concatenating the strings with '_' and append extension name
#         merged_col = rec_col + '_' + idx_col + ".wav"
        
#         self.dataset = merged_col.tolist()
#         self.load_dir = load_dir
#         self.transform = transform
        
    
#     def __len__(self):
#         """
#         Returns the length of the dataset.
        
#         Returns:
#             int: The number of input-output pairs in the dataset.
#         """
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         """
#         Returns a tuple (input_data, output_data) for the given index.

#         The function first checks if the provided index is a tensor, and if so, converts it to a list.
#         It then constructs the file path for the .wav file using the dataset attribute and the provided index.
#         The .wav file is loaded using torchaudio, and its data is normalized. If a transform is provided,
#         the data is transformed using the specified transform. Finally, the input_data and output_data are
#         set to the same data (creating a tuple), and the tuple is returned.

#         Args:
#         idx (int or torch.Tensor): The index of the desired data.

#         Returns:
#         tuple: A tuple containing input_data and output_data, both of which are the audio data
#                from the .wav file at the specified index.

#         Note: 
#         This function assumes that the class has the following attributes:
#         - self.load_dir (str): The directory containing the .wav files.
#         - self.dataset (list): A list of .wav file names.
#         - self.transform (callable, optional): An optional transform to apply to the audio data.
#         """
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         wav_name = os.path.join(self.load_dir,
#                                 self.dataset[idx])
        
#         data, sample_rate = torchaudio.load(wav_name, normalize=True)
#         if self.transform:
#             data = self.transform(data, sr=sample_rate)
        
#         # # Prepare for possible in-out discrepencies in the future
#         # input_data = data
#         # output_data = data
        
#         return data

# def collate_fn(xx):
#     # only working for one data at the moment
#     batch_first = True
#     x_lens = [len(x) for x in xx]
#     xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
#     return xx_pad, x_lens


# class MyTransform(nn.Module): 
#     def __init__(self, sample_rate, n_fft): 
#         super().__init__()
#         # self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=n_fft, n_mels=64)
#         # self.to_db = torchaudio.transforms.AmplitudeToDB()
#         # self.transform = torchaudio.transforms.MFCC(n_mfcc=13)
    
#     def forward(self, waveform, sr=16000): 
#         # extract mfcc
#         feature = torchaudio.compliance.kaldi.mfcc(waveform, sample_frequency=sr)

#         # add deltas
#         d1 = torchaudio.functional.compute_deltas(feature)
#         d2 = torchaudio.functional.compute_deltas(d1)
#         feature = torch.cat([feature, d1, d2], dim=-1)

#         # Apply normalization (CMVN)
#         eps = 1e-9
#         mean = feature.mean(0, keepdim=True)
#         std = feature.std(0, keepdim=True, unbiased=False)
#         # print(feature.shape)
#         # print(mean, std)
#         feature = (feature - mean) / (std + eps)

#         # mel_spec = self.transform(waveform)
#         # # mel_spec = self.to_db(mel_spec)
#         # mel_spec = mel_spec.squeeze()
#         # mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)
#         return feature
    


class SeqDatasetInfo(Dataset):
    """
    A PyTorch dataset that loads cutted wave files from disk and returns input-output pairs for
    training autoencoder. 
    
    Version 1: wav -> mfcc
    """
    
    def __init__(self, load_dir, load_control_path, transform=None):
        """
        Initializes the class by reading a CSV file and merging the "rec" and "idx" columns.

        The function reads the CSV file from the provided control path, extracts the "rec" and "idx" columns,
        and concatenates the values from these columns using an underscore. It then appends the ".wav" extension
        to each of the merged strings and converts the merged pandas Series to a list, which is assigned to
        the 'dataset' attribute of the class.

        Args:
        load_dir (str): The directory containing the files to load.
        load_control_path (str): The path to the CSV file containing the "rec" and "idx" columns.

        Attributes:
        dataset (list): A list of merged strings from the "rec" and "idx" columns, with the ".wav" extension.
        """
        control_file = pd.read_csv(load_control_path)
        control_file = control_file[control_file['n_frames'] > 400]
        control_file = control_file[control_file['duration'] <= 2.0]
        
        # Extract the "rec" and "idx" columns
        rec_col = control_file['rec'].astype(str)
        idx_col = control_file['idx'].astype(str).str.zfill(8)

        # Extract the "token" and "produced_segments" columns
        token_col = control_file['token'].astype(str)
        produced_segments_col = control_file['produced_segments'].astype(str)
        
        # Merge the two columns by concatenating the strings with '_' and append extension name
        merged_col = rec_col + '_' + idx_col + ".wav"
        
        self.dataset = merged_col.tolist()
        self.infoset = produced_segments_col.tolist()
        self.info_rec_set = rec_col.tolist()
        self.info_idx_set = idx_col.tolist()
        self.info_token_set = token_col.tolist()

        self.load_dir = load_dir
        self.transform = transform
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_name = os.path.join(self.load_dir,
                                self.dataset[idx])
        
        data, sample_rate = torchaudio.load(wav_name, normalize=True)
        if self.transform:
            data = self.transform(data, sr=sample_rate)
        
        info = self.infoset[idx]
        # extra info for completing a csv
        info_rec = self.info_rec_set[idx]
        info_idx = self.info_idx_set[idx]
        info_token = self.info_token_set[idx]
        
        return data, info, info_rec, info_idx, info_token

def collate_fn(data):
    # xx = data, aa bb cc = info_rec, info_idx, info_token
    xx, yy, aa, bb, cc = zip(*data)
    # only working for one data at the moment
    batch_first = True
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
    return xx_pad, x_lens, yy, aa, bb, cc


class MyTransform(nn.Module): 
    def __init__(self, sample_rate, n_fft): 
        super().__init__()
    
    def forward(self, waveform, sr=16000): 
        # extract mfcc
        feature = torchaudio.compliance.kaldi.mfcc(waveform, sample_frequency=sr)

        # add deltas
        d1 = torchaudio.functional.compute_deltas(feature)
        d2 = torchaudio.functional.compute_deltas(d1)
        feature = torch.cat([feature, d1, d2], dim=-1)

        # Apply normalization (CMVN)
        eps = 1e-9
        mean = feature.mean(0, keepdim=True)
        std = feature.std(0, keepdim=True, unbiased=False)

        feature = (feature - mean) / (std + eps)

        return feature