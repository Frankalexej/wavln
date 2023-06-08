class Sound_Proc: 
    """
    The Sound_Proc class provides methods for processing audio data.

    Attributes:
    - rate (int): the sample rate of the audio data (default 16000)

    Methods:
    - __init__(self): initializes the Sound_Proc instance with a default sample rate of 16000
    - get_rec_length(self, metadata): calculates the length of a recording in seconds based on the number of frames and sample rate in the metadata
    - time2frame(self, time_sec): converts a time in seconds to the corresponding number of audio frames
    - cut_rec(self, rec, flat_starts, flat_ends): segments a recording into individual phone recordings based on specified start and end times in seconds. Returns a list of phone recordings.
    """
    def __init__(self): 
        self.rate = 16000
        
    def get_rec_length(self, metadata): 
        return metadata.num_frames / metadata.sample_rate
    
    def time2frame(self, time_sec): 
        # Calculate the number of elapsed samples
        return int(round(time_sec * self.rate))
    
    def cut_rec(self, rec, flat_starts, flat_ends): 
        flat_start_frames = [self.time2frame(t) for t in flat_starts]
        flat_end_frames = [self.time2frame(t) for t in flat_ends]

        phone_list = [rec[:, start:end] for start, end in zip(flat_start_frames, flat_end_frames)]

        return phone_list