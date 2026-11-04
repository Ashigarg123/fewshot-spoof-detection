import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import sys
from torch import Tensor 

from data_utils import read_manifest

class Kshot_Dataset(Dataset):

    def __init__(self, dataset_path, audio_root=None):
        super().__init__()
        self.x = []
        self.y = []
        self.cut = 64600

        for path, label in read_manifest(dataset_path, audio_root):
            if label == "spoof":
                self.y.append(1)
            elif label == "bonafide":
                self.y.append(0)
            else:
                raise ValueError(f'unsupported binary label: {label}')
            self.x.append(path)
        print(f"Total samples loaded: {len(self.x)}")
    
    def __len__(self):
        return len(self.x)
    def pad(self, x, max_len=64600):
        # x is 1D here 
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        num_repeats = int(max_len/x_len) + 1 
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x
        
    def __getitem__(self, idx):
        utt_pth = self.x[idx]
        label = self.y[idx]
        signal, fs = librosa.load(utt_pth)
        if fs != 16000:
            signal = librosa.resample(signal, orig_sr=fs, target_sr=16000)
        #pad audio if not 4 seconds 
        new_signal = self.pad(signal, max_len=64600)
        x = Tensor(new_signal)
        
        # mel_spec = librosa.feature.melspectrogram(y=new_signal, **spec_params)
        # log_mel_spec = 20.0 / spec_params['power'] * np.log10(mel_spec + sys.float_info.epsilon)
        # log_mel_spec = torch.from_numpy(log_mel_spec) #torch.Size([128, 157])
        # log_mel_spec = log_mel_spec.unsqueeze(0)
        
        return x, label
    

    
