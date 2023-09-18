"""
This is the integrated dataset collection. 
Collate functions should be explicitly included in the definition of dataset as static method, 
instead independently defined outside so as to avoid overriding. 

"""
import pandas as pd
from torch.utils.data import Dataset
import torch
import torchaudio
import os
from torch.nn.utils.rnn import pad_sequence
from torch import nn

class SeqDataset(Dataset):
    """
    A PyTorch dataset that loads cutted wave files from disk and returns input-output pairs for
    training autoencoder. [wav -> mel]
    """
    
    def __init__(self, load_dir, load_control_path, transform=None):
        """
        Initializes the class by reading a CSV file and merging the "rec" and "idx" columns.

        Args:
        load_dir (str): The directory containing the files to load.
        load_control_path (str): The path to the CSV file containing the "rec" and "idx" columns.
        transform (Transform): when loading files, this will be applied to the sound data. 
        """
        control_file = pd.read_csv(load_control_path)
        control_file = control_file[control_file['n_frames'] > 400]
        control_file = control_file[control_file['duration'] <= 2.0]
        
        # Extract the "rec" and "idx" columns
        rec_col = control_file['rec'].astype(str)
        idx_col = control_file['idx'].astype(str).str.zfill(8)
        
        # Merge the two columns by concatenating the strings with '_' and append extension name
        merged_col = rec_col + '_' + idx_col + ".wav"
        
        self.dataset = merged_col.tolist()
        self.load_dir = load_dir
        self.transform = transform
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns a tuple (input_data, output_data) for the given index.

        The function first checks if the provided index is a tensor, and if so, converts it to a list.
        It then constructs the file path for the .wav file using the dataset attribute and the provided index.
        The .wav file is loaded using torchaudio, and its data is normalized. If a transform is provided,
        the data is transformed using the specified transform. Finally, the input_data and output_data are
        set to the same data (creating a tuple), and the tuple is returned.

        Note: 
        This function assumes that the class has the following attributes:
        - self.load_dir (str): The directory containing the .wav files.
        - self.dataset (list): A list of .wav file names.
        - self.transform (callable, optional): An optional transform to apply to the audio data.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_name = os.path.join(self.load_dir,
                                self.dataset[idx])
        
        data, sample_rate = torchaudio.load(wav_name, normalize=True)
        if self.transform:
            data = self.transform(data)
        
        return data

    @staticmethod
    def collate_fn(xx):
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens





class MelSpecTransform(nn.Module): 
    def __init__(self, sample_rate, n_fft=400, n_mels=64): 
        super().__init__()
        self.sample_rate = sample_rate
        n_stft = int((n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=n_fft)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
        self.grifflim = torchaudio.transforms.GriffinLim(n_fft=n_fft)
        
    
    def forward(self, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)

        """
        There should be normalization method here, 
        but for the moment we just leave it here, 
        later, consider PCEN
        """
        # # Apply normalization (CMVN)
        eps = 1e-9
        mean = mel_spec.mean(0, keepdim=True)
        std = mel_spec.std(0, keepdim=True, unbiased=False)
        # # print(feature.shape)
        # # print(mean, std)
        mel_spec = (mel_spec - mean) / (std + eps)

        # mel_spec = self.transform(waveform)
        # # mel_spec = self.to_db(mel_spec)
        # mel_spec = mel_spec.squeeze()
        # mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)
        return mel_spec
    
    def inverse(self, mel_spec): 
        mel_spec = mel_spec.permute(1, 0) # (L, F) -> (F, L)
        mel_spec = mel_spec.unsqueeze(0)  # restore from (F, L) to (channel, F, L)
        i_mel = self.inverse_mel(mel_spec)
        inv = self.grifflim(i_mel)
        return inv