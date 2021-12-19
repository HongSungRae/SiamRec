'''
Pre-train된 model을 불러옵니다
Test PoP 50곡을 들려주어서 50곡에 대한 vector를 모두 뽑아 Tensor shape [50,2048]을 만듭니다
Tensor를 tSNE package에 넣어서 결과를 뽑습니다
model이 있는 path에 이미지를 저장합니다
# https://pypi.org/project/tsne-torch/
'''

# from library
import torch
import torch.nn as nn
import argparse
import os
from glob import glob
from tsne_torch import TorchTSNE as TSNE
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
import sys

# from local
from simsiam import Siamusic, TestEncoder
from dataset import JsonAudio


# argparser
parser = argparse.ArgumentParser(description='Siamusic-tSNE')

parser.add_argument('--save_path', default='', type=str,
                    help='실험할 모델이 있는 위치. 이 위치에 tSNE 결과물도 저장됩니다.')
parser.add_argument('--backbone', default='resnet50', type=str,
                    choices=['resnet50','resnet101','resnet152','transformer'],
                    help='백본')
parser.add_argument('--dim', default=2048, type=int,
                    help='모델의 아웃풋 dim')
parser.add_argument('--nhead', default=4, type=int,
                    help='transformer의 헤드 수')
parser.add_argument('--iter', default=30, type=int,
                    help='노래를 몇번 들려줄거냐?')
args = parser.parse_args()


if args.save_path == '' or (not os.path.exists(args.save_path)):
    raise ValueError('Test할 모델이 있는 위치가 불명확합니다')



def get_tSNE(iter=args.iter):
    '''
    Pre-train된 model을 불러옵니다
    Test PoP 50곡을 들려주어서 50곡에 대한 vector를 모두 뽑아 Tensor shape [50,2048]을 만듭니다
    Tensor를 tSNE package에 넣어서 결과를 뽑습니다
    model이 있는 path에 이미지를 저장합니다
    # https://pypi.org/project/tsne-torch/
    '''
    # 로더 정의
    pop_path = 'D:/Siamusic/dataset/TestPOP_json'
    dataset = JsonAudio(pop_path,48000)
    loader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=False)
    song_list = dataset.data_list

    # 처음부터 조사한다면
    if 'vector.json' not in os.listdir(args.save_path):
        # 모델 불러오기
        pth_file = ''
        for file in glob(args.save_path+'/*.pth'):
            pth_file = file
        pth = torch.load(pth_file)

        # key 바꿀 필요가 있을 때
        keys = list(pth.keys())
        for key in keys:
            pth[key.replace('module.','')] = pth.pop(key)

        try:
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                model = Siamusic(backbone=args.backbone,
                        dim=args.dim,
                        nhead=args.nhead).cuda()
                model = nn.DataParallel(model).cuda()
                model.load_state_dict(pth)
                print('case 1')
            except:
                os.environ['CUDA_VISIBLE_DEVICES'] = '1'
                model = Siamusic(backbone=args.backbone,
                        dim=args.dim,
                        nhead=args.nhead).cuda()
                model = nn.DataParallel(model).cuda()
                model.load_state_dict(pth) 
                print('case 2')
        except:
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                model = Siamusic(backbone=args.backbone,
                        dim=args.dim,
                        nhead=args.nhead).cuda()
                model.load_state_dict(pth) 
                print('case 3')
            except:
                os.environ['CUDA_VISIBLE_DEVICES'] = '1'
                model = Siamusic(backbone=args.backbone,
                        dim=args.dim,
                        nhead=args.nhead).cuda()
                model.load_state_dict(pth)
                print('case 4')
        torch.cuda.set_device(0) # 0 이 1번 , 1이 0번 GPU
        test_model = TestEncoder(backbone=args.backbone,
                                dim=args.dim,
                                nhead=args.nhead).cuda()
        test_model = nn.DataParallel(test_model,device_ids=[0,1]).cuda()
        model = model.cuda()
        # test_model.encoder = model.module.encoder
        test_model.encoder = model.encoder
        del(model)

        # 음악을 1곡씩 불러와서 forward 후 평균내기
        pop_vector = torch.zeros((50,2048))
        test_model.eval()
        music_list = os.listdir('./dataset/TestPOP_json')
        for i in tqdm(range(iter),desc='음악 듣는중...'):
            for idx,audio in enumerate(loader):
                audio = audio.float().cuda() # [B, 1, 48000]
                pred = test_model(audio)
                pred = torch.squeeze(pred,dim=0)
                pop_vector[idx] += pred.detach().cpu()/iter
                del(pred)
        vector = pop_vector.detach().cpu().clone() # [50, 2048]

        # vector만 저장
        vector_dict = {'vector':vector.tolist()}
        with open(args.save_path + '/vector.json', 'w') as f:
            json.dump(vector_dict, f)
    
    # 이미 조사결과가 있다면
    else:
        with open(args.save_path + '/vector.json', 'r') as f:
            v = json.load(f)['vector']
            vector = torch.Tensor(v)

    # tSNE로 차원축소
    try:
        X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(vector)  # returns shape (n_samples, 2)
    except:
        # 오류나는 코드가 library에서 출발한다. 코드 까보니까 Index 설정이 뭐 하나 잘못되어있다
        dummy = torch.zeros(50,1)
        vector = torch.cat((vector,dummy),dim=-1)
        X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(vector)  # returns shape (n_samples, 2)

    # 아티스트별로 그룹 만들기
    artist_list = [i.split('-')[0] for i in song_list] # 50
    a = ['Taeyeon', 'Nell', 'Green Day ', 'BTS ', '2NE1', 'One Direction', 'IU', 'AKMU', 'Oasis', 'Beenzino']
    artists = [a[4],a[7],a[9],a[3],a[2],a[6],a[1],a[8],a[5],a[0]] # 10
    X_emb_list = X_emb.tolist()
    data = {}
    for i,artist in enumerate(artist_list):
        if artist not in data.keys():
            data[artist] = []
        data[artist].append(X_emb_list[i])
    else:
        for key in data.keys():
            data[key] = np.array(data[key])

    # figure 및 axis 설정
    fig, ax = plt.subplots()
    markers = ['o', 'v', '^', '<', '>', 's',  '*', 'd', 'P', 'X']
    sizes = [40 for i in range(10)]
    widths = [0.5 for i in range(10)]
    
    # 그리기 - 노래별로 plot
    '''
    https://frhyme.github.io/python-lib/text-in-matplotlib/
    https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
    https://sarah0518.tistory.com/99
    '''
    idx = 0
    for i,artist in enumerate(artists):
        X = data[artist][:,0]
        Y = data[artist][:,1]
        ax.scatter(X, Y, marker=markers[i], label=artists[i],
                    s=sizes[i], edgecolor='k', linewidth=widths[i])
        for song_idx in range(len(X)):
            plt.text(X[song_idx]+0.1,Y[song_idx],song_list[idx].split('.')[0].split('-')[1],fontsize=5)
            idx += 1
            # plt.text(X[song_idx]+0.1,Y[song_idx],song_idx)
            # ax.annotate([0,1,2,3,4], (X[song_idx], Y[song_idx]))
            # ax.scatter(X[song_idx], Y[song_idx], marker=markers[i], label=artists[i],
            #         s=sizes[i], edgecolor='k', linewidth=widths[i])
    ax.legend(loc='best')
    plt.tight_layout()
    plt.title(args.save_path.split('/')[1],fontsize=20)

    # 저장하기
    plt.savefig(args.save_path+'/tSNE_song.png', format='png',dpi=300)
    plt.clf()

    # 그리기 - 아타스트별로 plot
    fig, ax = plt.subplots()
    for i,artist in enumerate(artists):
        X = sum(data[artist][:,0])/5
        Y = sum(data[artist][:,1])/5
        ax.scatter(X, Y, marker=markers[i], label=artists[i],
                    s=sizes[i], edgecolor='k', linewidth=widths[i])
        plt.text(X+0.1,Y,artist,fontsize=5)
    
    plt.title(args.save_path.split('/')[1],fontsize=20)
    plt.savefig(args.save_path+'/tSNE_artist.png', format='png',dpi=300)
    plt.clf()




if __name__ == '__main__':
    get_tSNE()