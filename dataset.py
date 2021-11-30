import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import librosa
# from scipy.io.wavfile import read



class MP3Audio(Dataset):
    '''
    mp4 audio는 학습된 모델의 train/val단계에서 사용됩니다
    '''
    def __init__(self,split=None,input_length=48000,type='wav'):
        if split not in ['train','validation']:
            raise ValueError()
        self.split = split
        self.input_length = input_length
        self.dir = './data/' + type + '_audio'
        self.audios = os.listdir(self.dir)
        if split == 'train':
            self.data_list = self.audios[0:int(len(self.audios)*0.85)]
        else:
            self.data_list = self.audios[int(len(self.audios)*0.85):]

    def __len__(self):
        return len(self.data_list)

    def get_waveform(self,data_path):#22050
        waveform,_ = librosa.load(data_path,sr=22050,duration=60)
        waveform = np.array(waveform,dtype=float)
        random_idx = np.random.randint(low=0, high=int(waveform.shape[0] - self.input_length))
        waveform = waveform[random_idx:random_idx+self.input_length] # extract 48000 sequence
        audio = np.expand_dims(waveform, axis = 0) # expand to [1,48000]
        return audio

    def __getitem__(self, idx):
        data_path = self.dir + '/' + self.data_list[idx]
        waveform = self.get_waveform(data_path)
        return waveform.astype(np.float32)





class MP4Audio(Dataset):
    '''
    mp4 audio는 학습된 모델의 test단계에서 사용됩니다
    '''
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass



if __name__ == '__main__':
    # mp3
    mp3_data = MP3Audio('validation')
    mp3_dataloader = DataLoader(mp3_data,batch_size=8,drop_last=True)
    mp3_x = next(iter(mp3_dataloader))
    print(f'mp3_x : {mp3_x.shape}')

    # # mp4
    # mp4_data = MP4Audio()
    # mp4_dataloader = DataLoader(mp4_data,batch_size=16,drop_last=True)
    # mp4_x, mp4_y= next(iter(mp4_dataloader))
    # print(f'mp4_x : {mp4_x.shape} | mp4_y : {mp4_y.shape}')