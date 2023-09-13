import numpy as np

class ResHandler: 
    def __init__(self, path="", data=None):
        self.path = path
        self.data = data

    def save(self): 
        try:
            np.save(self.path, self.data)
        except Exception as e:
            print(f"Error saving data: {str(e)}")

    def read(self): 
        try:
            self.data = np.load(self.path)
        except Exception as e:
            print(f"Error loading data: {str(e)}")