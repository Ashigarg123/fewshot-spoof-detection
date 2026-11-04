import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import sys
from torch import Tensor 
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange

from data_utils import read_manifest

class Kshot_Dataset(Dataset):

    def __init__(self, dataset_path, args, algo, is_train=True, label_map=None,
                 audio_backend='librosa'):
        super().__init__()
        self.x = []
        self.y = []
        self.is_train = is_train 
        self.cut = 64600
        self.args = args
        self.algo = algo
        self.audio_backend = audio_backend
        self.mp = label_map or {
            "bonafide": 0,
            "A01": 1,
            "A02": 2,
            "A03": 3,
            "A04": 4,
            "A05": 5,
            "A06": 6,
        }


        for path, label in read_manifest(dataset_path, args.audio_root):
            if label in self.mp:
                self.y.append(self.mp[label])
            else:
                raise ValueError(f'unsupported multi-class label: {label}')
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
        if self.audio_backend == 'torchaudio':
            signal, fs = torchaudio.load(utt_pth)
            if fs != 16000:
                signal = torchaudio.functional.resample(signal, fs, 16000)
            signal = signal.squeeze().numpy()
        else:
            signal, fs = librosa.load(utt_pth)
            if fs != 16000:
                signal = librosa.resample(signal, orig_sr=fs, target_sr=16000)
        #pad audio if not 4 seconds 
        Y=process_Rawboost_feature(signal,16000,self.args,self.algo)
        new_signal = self.pad(Y, max_len=64600)
        x = Tensor(new_signal)
        
        # mel_spec = librosa.feature.melspectrogram(y=new_signal, **spec_params)
        # log_mel_spec = 20.0 / spec_params['power'] * np.log10(mel_spec + sys.float_info.epsilon)
        # log_mel_spec = torch.from_numpy(log_mel_spec) #torch.Size([128, 157])
        # log_mel_spec = log_mel_spec.unsqueeze(0)
        
        return x, label
    

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature






    
