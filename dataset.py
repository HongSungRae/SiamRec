import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import librosa
import json
# from scipy.io.wavfile import read



class MP3Audio(Dataset):
    '''
    # https://nbviewer.org/github/mdeff/fma/blob/outputs/usage.ipynb
    mp4 audio는 학습된 모델의 train단계에서 사용됩니다.
    '''
    def __init__(self,input_length=48000,type='small'):
        if type not in ['small','medium']:
            raise ValueError('samll or medium')
        self.input_length = input_length
        self.dir = 'D:/Siamusic/dataset/fma_' + type
        self.folders = next(os.walk(self.dir))[1]
        self.df = pd.DataFrame(columns=['path'])
        for folder in self.folders:
            PATH = self.dir + '/' + folder
            for music in next(os.walk(PATH))[2]:
                self.df = self.df.append({"path": PATH+'/'+music}, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def get_waveform(self,data_path):#22050
        waveform,_ = librosa.load(data_path,sr=22050,duration=30)
        waveform = np.array(waveform,dtype=float)
        random_idx = np.random.randint(low=0, high=int(waveform.shape[0] - self.input_length))
        waveform = waveform[random_idx:random_idx+self.input_length] # extract 48000 sequence
        audio = np.expand_dims(waveform, axis = 0) # expand to [1,48000]
        return audio

    def __getitem__(self, idx):
        data_path = self.df['path'][idx]
        waveform = self.get_waveform(data_path)
        return waveform.astype(np.float32)




class JsonAudio(Dataset):
    '''
    mp3를 python에서 부르는 것은 매우 느립니다.
    json으로 변환된 음악을 불러오는 class입니다.
    '''
    def __init__(self,data_dir,input_length=48000):
        self.data_dir = data_dir
        self.data_list = os.listdir(data_dir)
        self.input_length = input_length

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        with open(self.data_dir+'/'+self.data_list[idx], 'r') as f:
            waveform = np.array(json.load(f)['audio'],dtype=float)
        random_idx = np.random.randint(low=0, high=int(waveform.shape[-1] - self.input_length))
        waveform = waveform[0][random_idx:random_idx+self.input_length]
        audio = np.expand_dims(waveform, axis = 0) # expand to [1,48000]
        return audio




class TestJsonAudio(Dataset):
    '''
    학습된 모델의 test단계에서 사용됩니다
    '''
    def __init__(self,df,data_dir,input_length=48000):
        self.df = df
        self.data_dir = './data/json_audio'
        self.input_length = input_length
        self.error_term = ['/','\"','<','>','\\','|',':','*','?']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid,artist,track,url = self.df.iloc[idx]
        song = artist + '-' + track
        for item in song:
            if item in self.error_term:
                song = song.replace(item,'^')
        try:
            data = self.data_dir + '/' + song + '.mp4.json'
            with open(data, 'r') as f:
                waveform = np.array(json.load(f)['audio'],dtype=float)
        except:
            data = self.data_dir + '/' + song + '.mp3.json'
            with open(data, 'r') as f:
                waveform = np.array(json.load(f)['audio'],dtype=float)
        random_idx = np.random.randint(low=0, high=int(waveform.shape[-1] - self.input_length))
        waveform = waveform[0][random_idx:random_idx+self.input_length]
        audio = np.expand_dims(waveform, axis = 0) # expand to [1,48000]
        return audio



if __name__ == '__main__':
    # mp3
    mp3_data = MP3Audio()
    mp3_dataloader = DataLoader(mp3_data,batch_size=8,drop_last=True)
    mp3_x = next(iter(mp3_dataloader))
    print(f'mp3_x : {mp3_x.shape}')

    # # mp4
    # mp4_data = MP4Audio()
    # mp4_dataloader = DataLoader(mp4_data,batch_size=16,drop_last=True)
    # mp4_x, mp4_y= next(iter(mp4_dataloader))
    # print(f'mp4_x : {mp4_x.shape} | mp4_y : {mp4_y.shape}')