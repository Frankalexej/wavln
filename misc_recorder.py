import pickle

# Define recorders of training hists, for ease of extension
class Recorder: 
    def __init__(self, IOPath): 
        self.record = []
        self.IOPath = IOPath

    def save(self): 
        pass
    
    def append(self, content): 
        self.record.append(content)
    
    def get(self): 
        return self.record
    

class LossRecorder(Recorder): 
    def read(self): 
        # only used by loss hists 
        with open(self.IOPath, 'rb') as f:
            self.record = pickle.load(f)
    
    def save(self): 
        with open(self.IOPath, 'wb') as file:
            pickle.dump(self.record, file)


class HistRecorder(Recorder):     
    def save(self): 
        with open(self.IOPath, "a") as txt:
            txt.write("\n".join(self.record))
    
    def print(self, content): 
        self.append(content)
        print(content)