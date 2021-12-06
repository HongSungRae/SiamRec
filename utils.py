from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
from matplotlib import pyplot as plt
from collections.abc import Iterable
import sounddevice as sd
from torch.utils.data import DataLoader
import pygame
import pygame.mixer
import winsound
import IPython.display as ipd
import time
from pydub import AudioSegment
import os
from tqdm import tqdm
import json
import librosa
from dataset import MP3Audio, MP4Audio



# from_dir에 있는 모든 audio를 duration(초)만큼만 load합니다
# load된 array를 key=노래제목, value = list(array) json으로
# to_dir에 저장합니다
def mp3_to_json(from_dir,to_dir,duration=29,sr=22050):
    music_list = next(os.walk(from_dir))[2]
    for music in tqdm(music_list):
        try:
            audio,_ = librosa.load(from_dir+'/'+music,sr=sr,duration=duration)
            audio = np.array(audio,dtype=float)
            audio_dict = {'audio' : audio.tolist()}
            os.chdir(to_dir)
            with open(music+'.json', 'w') as fp:
                json.dump(audio_dict, fp)
                os.chdir('./')
        except:
            print(f'{music}은(는) 변환되지 않습니다.')




# from_dir의 모든 mp3 file을 
# wav format으로 바꾸어 to_dir로 보냅니다
def mp3_to_wav(from_dir,to_dir):
    mp3_list = os.listdir(dir)
    for mp3 in tqdm(mp3_list):                                                                      
        src = from_dir + '/' + mp3
        if '.mp3' in src:
            dst = to_dir + '/' + mp3.replace('.mp3','.wav')
        else:
            dst = to_dir + '/' + mp3 + '.wav'
        audSeg = AudioSegment.from_mp3(src)
        audSeg.export(dst, format="wav")



def listen(audio,fs=48000):
    # 'audio' is waveform which type is torch.tensor
    # audio shape : [fs]
    sd.play(audio, 22050, blocking=True)


def listen_raw(data_path):
    if 'MTA' in data_path:
        waveform = np.load(data_path.replace(".mp3",".npy"))
        ipd.Audio(waveform, rate=16000)
        pygame.mixer.init()
        pygame.mixer.music.load(data_path)
        pygame.mixer.music.play()
    elif 'GTZAN' in data_path:
        winsound.PlaySound(data_path, winsound.SND_FILENAME)



###########
# logging #
###########
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0  # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val != None:  # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val ** 2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2 / self.count - self.avg ** 2)
        else:
            pass


class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try:
            return len(self.read())
        except:
            return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.', v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log

def draw_curve(work_dir, train_logger, test_logger):
        train_logger = train_logger.read()
        test_logger = test_logger.read()
        epoch, train_loss = zip(*train_logger)
        epoch,test_loss = zip(*test_logger)

        plt.plot(epoch, train_loss, color='blue', label="Train Loss")
        plt.plot(epoch, test_loss, color='red', label="Test Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(work_dir + '/loss_curve.png')
        plt.close()




if __name__ == '__main__':
    ## Test listen()
    # fs = 160000 #48000
    # bs = 8
    # start = time.time()
    # mp3_data = MP3Audio('validation',fs)
    # mp3_dataloader = DataLoader(mp3_data,batch_size=bs,drop_last=True)
    # mp3_x = next(iter(mp3_dataloader))
    # print(f'Batch Size : {bs} | Time Duration : {time.time()-start}')
    # listen(mp3_x[3,0],fs)
    # time.sleep(1)

    ## Test listen_raw()
    # EnterSandman = './dataset/GTZAN/genres_original/metal/metal.00033.wav'
    # SnoopDogg = './dataset/GTZAN/genres_original/hiphop/hiphop.00033.wav'
    # listen_raw(SnoopDogg)
    # listen_raw(EnterSandman)

    ## Mp3 to Wav : Recommended. python reads Wav faster than Mp3 format
    # mp3_to_wav('./data/train_audio')

    ## Mp3 to json
    pass