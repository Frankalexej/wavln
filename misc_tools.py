# Here we put some functions / classes as tool
# Let's make the code better structured! 
import os
import datetime
import pandas as pd
from IPython.display import Audio, display

class PathUtils: 
    @staticmethod
    def path_exist(path): 
        return os.path.exists(path)
    
    @staticmethod
    def path_isdir(path): 
        return os.path.isdir(path)

    @staticmethod
    def mk(dir): 
        os.makedirs(dir, exist_ok = True)



class ARPABET: 
    @staticmethod
    def list_vowels(ah=False): 
        if ah: 
            return ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY', 'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX']
        else:
            return ['AA', 'AE', 'AO', 'AW', 'AX', 'AXR', 'AY', 'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX']
    
    @staticmethod
    def list_consonants(): 
        return [
            'B', 'CH', 'D', 'DH', 'DX', 'EL', 'EM', 'EN', 'F', 'G', 'HH', 'H', 'JH', 'K', 'L', 'M', 'N',
            'NX', 'NG', 'P', 'Q', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH'
        ]
    
    @staticmethod
    def intersect_lists(list1, list2):
        """
        Keep the order as in list1
        """
        return [x for x in list1 if x in list2]

    @staticmethod
    def is_vowel(arpabet_transcription):
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY', 'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX']
        
        if arpabet_transcription in vowels:
            return True
        else:
            return False
    
    @staticmethod
    def is_consonant(arpabet_transcription):
        consonants = [
            'B', 'CH', 'D', 'DH', 'DX', 'EL', 'EM', 'EN', 'F', 'G', 'HH', 'H', 'JH', 'K', 'L', 'M', 'N',
            'NX', 'NG', 'P', 'Q', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH'
        ]
        
        if arpabet_transcription in consonants:
            return True
        else:
            return False
    
    @staticmethod
    def vowel_consonant(arpabet_transcription): 
        if ARPABET.is_vowel(arpabet_transcription): 
            return "vowel"
        elif ARPABET.is_consonant(arpabet_transcription): 
            return "consonant"
        else: 
            return "nap"
    
    @staticmethod
    def vowel_consonant_num(arpabet_transcription): 
        if ARPABET.is_vowel(arpabet_transcription): 
            return 1
        elif ARPABET.is_consonant(arpabet_transcription): 
            return 0
        else: 
            return -1
    
    VOWEL_CODE = 1
    CONSONANT_CODE = 0
    NAP_CODE = -1
        
class MyAudio: 
    @staticmethod
    def play_audio_torch(waveform, sample_rate):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        if num_channels == 1:
            display(Audio(waveform[0], rate=sample_rate))
        elif num_channels == 2:
            display(Audio((waveform[0], waveform[1]), rate=sample_rate))
        else:
            raise ValueError("Waveform with more than 2 channels are not supported.")
        
    @staticmethod
    def play_audio_np(waveform, sample_rate):
        display(Audio(waveform[0], rate=sample_rate))


class AudioCut: 
    @staticmethod
    def time2frame(time_sec, rate): 
        # Calculate the number of elapsed samples
        return int(round(time_sec * rate))
    
    @staticmethod
    def idx2text(idx, fill_num): 
        return str(idx).zfill(fill_num)
    
    @staticmethod
    def cut_name_gen(name, idx, fill_num, bare=False): 
        if bare: 
            return name + "-" + AudioCut.idx2text(idx, fill_num)
        return name + "-" + AudioCut.idx2text(idx, fill_num) + ".flac"
    
    @staticmethod
    def solve_name(name): 
        # bare name, no extension
        # 19-198-0037 (example)
        return tuple(name.split("-"))
    
    @staticmethod
    def record2speaker(record): 
        # record is a row
        sentence_name = record["file"]
        speaker, rec, sentence = AudioCut.solve_name(sentence_name)
        return speaker

    @staticmethod
    def record2filepath(record): 
        # record is a row
        sentence = record["file"]
        idx = AudioCut.idx2text(record["id"], fill_num=4)

        return sentence.replace("-", "/") + "/" + sentence + "-" + idx + ".flac"
    
    @staticmethod
    def wordrecord2filepath(record): 
        # record is a row
        sentence = record["file"]
        try: 
            idx = int(record["word_id"])
        except:
            idx = "xxxx"
        idx = AudioCut.idx2text(idx, fill_num=4)

        return sentence.replace("-", "/") + "/" + sentence + "-" + idx + ".flac"
    
    @staticmethod
    def wordrecord2wuid(record): 
        sentence = record["file"]
        try: 
            idx = int(record["word_id"])
        except:
            idx = "xxxx"
        idx = AudioCut.idx2text(idx, fill_num=4)

        return sentence + "-" + idx
    
    @staticmethod
    def syllablerecord2filepath(record): 
        # record is a row
        sentence = record["file"]
        try: 
            idx = int(record["syllable_id"])
        except:
            idx = "xxxx"
        idx = AudioCut.idx2text(idx, fill_num=4)

        return sentence.replace("-", "/") + "/" + sentence + "-" + idx + ".flac"
    
    @staticmethod
    def syllablerecord2wuid(record): 
        sentence = record["file"]
        try: 
            idx = int(record["syllable_id"])
        except:
            idx = "xxxx"
        idx = AudioCut.idx2text(idx, fill_num=4)

        return sentence + "-" + idx
    
    @staticmethod
    def filename_id2filepath(filename, idx): 
        # record is a row
        sentence = filename
        idx = AudioCut.idx2text(idx, fill_num=4)

        return sentence.replace("-", "/") + "/" + sentence + "-" + idx + ".flac"

# START
def get_timestamp():
    # for model save
    return datetime.datetime.now().strftime("%m%d%H%M%S")   # timestamp ignores year and second, not really needed. 

# Calculate mean and variance online (i.e., without storing all data points)
class OnlineMeanVariance:
    def __init__(self):
        self.n = 0       # Number of data points
        self.mean = 0.0   # Running mean
        self.M2 = 0.0     # Sum of squared differences from the mean (for variance calculation)
    
    def update(self, x):
        """Add a new data point and update mean and variance"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    def get_mean(self):
        """Return the current mean"""
        return self.mean
    
    def get_variance(self):
        """Return the current variance"""
        if self.n < 2:
            return float('nan')  # Variance is undefined for fewer than 2 data points
        return self.M2 / (self.n - 1)  # Use (n-1) for sample variance

    def get_population_variance(self):
        """Return the population variance (divide by n instead of n-1)"""
        if self.n < 1:
            return float('nan')  # Variance is undefined for no data points
        return self.M2 / self.n  # Use n for population variance