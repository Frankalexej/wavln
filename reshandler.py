import numpy as np
import pickle
import os

class EncoderResHandler: 
    def __init__(self, data_dir, info_dir, file_prefix="", data=None, info=None):
        self.data_dir = data_dir
        self.info_dir = info_dir
        self.file_prefix = file_prefix
        self.data = data
        self.info = info

    def save(self): 
        try:
            np.save(os.path.join(self.data_dir, self.file_prefix + ".npy"), self.data)
        except Exception as e:
            print(f"Error saving data: {str(e)}")

        try:
            with open(os.path.join(self.info_dir, self.file_prefix + ".info"), 'wb') as file:
                pickle.dump(self.info, file)
        except Exception as e:
            print(f"Error saving info: {str(e)}")

    def read(self): 
        try:
            self.data = np.load(os.path.join(self.data_dir, self.file_prefix + ".npy"))
        except Exception as e:
            print(f"Error loading data: {str(e)}")

        try:
            with open(os.path.join(self.info_dir, self.file_prefix + ".info"), 'rb') as file:
                self.info = pickle.load(file)
        except Exception as e:
            print(f"Error loading info: {str(e)}")