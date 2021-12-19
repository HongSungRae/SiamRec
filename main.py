# library
from numpy.random import choice, shuffle
from torch.nn.modules.activation import Threshold
from torch.utils.data import DataLoader
import os
import torch
import argparse
import torch.optim as optim
import json
import torch.nn as nn
from torch.optim import lr_scheduler
from torchmetrics.classification import accuracy
from tqdm import tqdm
from torchmetrics import F1,AUROC,Accuracy,Recall
import time
import sys
import pandas as pd
from glob import glob
import random

# local
from utils import *
from dataset import JsonAudio, TestJsonAudio
from loss import SiamusicLoss
from augmentation import sungrae_pedal, random_mix, image_augmentation
from simsiam import Siamusic, TestEncoder, Siamusic_Image
from metric import get_reward


parser = argparse.ArgumentParser(description='Siamusic')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='save path')
parser.add_argument('--backbone', default='ResNet34', type=str,
                    help='backbone network for simsiam',
                    choices=['resnet50','resnet101','resnet152','transformer'])
parser.add_argument('--dim', default=2048, type=int,
                    help='output dimension')
parser.add_argument('--nhead', default=4, type=int,
                    help='the number of transformer heads',choices=[1,2,4])
parser.add_argument('--input_length', default=48000, type=int,
                    help='input length')
parser.add_argument('--fma', default='samll', type=str,
                    help='어떤 데이터셋으로 pre-train 할건가?',
                    choices=['medium','small'])
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--optim', default='adam', type=str,
                    help='optimizer', choices=['sgd','adam','adagrad'])
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.00001, type=float,
                    help='weight_decay')
parser.add_argument('--epochs', default=100, type=int,
                    help='train epoch')
parser.add_argument('--augmentation', default='pedalboard', type=str,
                    help='train epoch',choices=['pedalboard','randommix','image'])
parser.add_argument('--patchs', default=12, type=int,
                     help='ramdom mix augmentation patchs')
parser.add_argument('--from_scratch', default=False, type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='How To Make TRUE? : --from_scratch, Flase : --no-from_scratch')
parser.add_argument('--threshold', default=0.5, type=float,
                    help='MTA 에서 confidence가 얼마 이상이면 1로 예측했다고 할 것인가?')
parser.add_argument('--n_show', default=15, type=int,
                    help='Test에서 플레이리스트 당 몇곡 보여줄 것 인가?')

parser.add_argument('--comment', default='', type=str,
                    help='Any comment you want')								
parser.add_argument('--gpu_id', default='1', type=str,
                    help='How To Check? : cmd -> nvidia-smi')
args = parser.parse_args()
start = time.time()



# siam model pretrain
def siam_train(model, trn_loader, criterion, optimizer, epoch, num_epoch, train_logger):
    model.train()
    train_loss = AverageMeter()
    for i, audio in enumerate(trn_loader):
        audio = audio.float()
        if args.augmentation == 'pedalboard':
            x1, x2 = sungrae_pedal(audio), sungrae_pedal(audio)
        elif args.augmentation == 'randommix':
            x1, x2 = random_mix(audio,args.patchs), random_mix(audio,args.patchs)
        elif args.augmentation == 'image':
            x1, x2 = image_augmentation(audio.numpy()), image_augmentation(audio.numpy()) 
        x1, x2 = x1.cuda(), x2.cuda()
        p1, z2, p2, z1 = model(x1,x2) # backbone(+projection) + predictor
        loss = criterion(p1,z2,p2,z1)

        train_loss.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            print('Epoch : [{0}/{1}] [{2}/{3}]  Train Loss : {loss:.4f}'.format(
                epoch, num_epoch, i, len(trn_loader), loss=loss))
    train_logger.write([epoch, train_loss.avg])

    

# downstream tast test
def test(vector,n_show=15,n_test=10):
    print("=================== Test Start ====================")
    reward = AverageMeter()

    df = pd.read_csv('./data/spotify_million_playlist/url.csv')
    df_show = pd.DataFrame(df.columns) # 15곡 split
    df_candidates = pd.DataFrame(df.columns) # 5곡 split
    for tst in range(n_test): # 10회 반복실험
        # 평균 Tensor
        show = torch.zeros((n_show*100,args.dim))
        candidates = torch.zeros(((20-n_show)*100,args.dim))

        # Shuffle & Split
        show_idx = []
        candidates_idx = []
        for i in range(100):
            temp = [i*20+j for j in range(20)]
            random.shuffle(temp)
            show_idx += temp[0:n_show]
            candidates_idx += temp[n_show:]
        for i,idx in enumerate(show_idx):
            show[i] = vector[idx]
        for i,idx in enumerate(candidates_idx):
            candidates[i] = vector[idx]

        r = get_reward(show,candidates)
        reward.update(r)

    print(f'Reward Score : {reward.avg:.2f}±{reward.std:.2f}')
    print("=================== Test End ====================")




def main():
    
    # pre-training or fine-tuning
    if args.from_scratch: ## pre-training
        print('스크래치부터 학습되고 Test까지 진행합니다.')

        # define environment
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        model = Siamusic(backbone=args.backbone,
                     dim=args.dim,
                     nhead=args.nhead).cuda()

        # save path
        save_path=args.save_path+'_'+args.backbone+'_'+args.augmentation+'_'+args.optim+'_'+args.fma
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # dataset loading
        dataset_path = 'D:/Siamusic/dataset/fma_'+args.fma+'_json'
        train_dataset = JsonAudio(dataset_path,args.input_length)
        test_dataset = JsonAudio(data_dir=dataset_path)
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True,drop_last=True)
        test_loader = DataLoader(test_dataset,batch_size=20,num_workers=2)
        print('=== DataLoader R.e.a.d.y ===')
        print(f'Train dataset : {len(train_dataset)}')
        print(f'Test dataset : {len(test_dataset)}')

        # define criterion
        criterion = SiamusicLoss().cuda()
        if args.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        milestones = [int(args.epochs/3),int(args.epochs/2)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)

        # logger
        train_logger = Logger(os.path.join(save_path, 'train_loss.log'))
        test_logger = Logger(os.path.join(save_path, 'train_loss.log'))

        # 학습시작
        for epoch in tqdm(range(1,args.epochs+1)):
            siam_train(model, train_loader, criterion ,optimizer, epoch, args.epochs, train_logger)
            scheduler.step()
            # 모델저장
            if epoch == args.epochs :
                path = '{0}/{1}_{2}_{3}.pth'.format(save_path,
                                                    args.backbone,
                                                    args.augmentation,
                                                    epoch)
                torch.save(model.state_dict(), path)    
        draw_curve(save_path, train_logger, train_logger)

        # 테스트시작
        test_model = TestEncoder(backbobe=args.backbone,
                                 dim=args.dim,
                                 nhead=args.nhead).cuda()
        test_model.encoder = model.encoder
        test(test_model, test_loader, test_logger, args)
    
    else: ## Test
        '''
        # 학습모델 넣기 참고용
        https://justkode.kr/deep-learning/pytorch-save
        https://tutorials.pytorch.kr/beginner/saving_loading_models.html
        '''
        print('학습된 Siam 모델을 불러와서 Test를 진행합니다.')

        # 순서
        '''
        1. 모델 불러와서 주입
        2. 2000곡을 20번 forward한 [2000,2048] 짜리 벡터 생성하여 저장
        3. 15-5개 찢는 코드를 만들어 30회 반복 테스트
        4. 테스트 점수 출력 및 저장
        '''
        # 모델 불러오기
        torch.cuda.set_device(0) # 0 이 1번 , 1이 0번 GPU
        print(f'Current GPU Device : {torch.cuda.current_device()}')

        PATH = './exp_'+args.backbone+'_'+args.augmentation+'_'+args.optim+'_small'

        if 'vector.json' not in os.listdir(PATH):
            pth_file = ''
            for file in glob(PATH+'/*.pth'):
                pth_file = file
            pth = torch.load(pth_file)

            # key 바꿀 필요가 있을 때
            keys = list(pth.keys())
            # for key in keys:
            #     pth[key.replace('module.','')] = pth.pop(key)
            
            if args.augmentation == 'image':
                # sys.exit('아직 구현중')
                try:
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    test_model = Siamusic_Image(backbone=args.backbone,
                                           dim=args.dim,
                                           nhead=args.nhead).cuda()
                    test_model = nn.DataParallel(test_model).cuda()
                    test_model.load_state_dict(pth)
                    print('case 0')
                except:
                    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
                    test_model = Siamusic_Image(backbone=args.backbone,
                                           dim=args.dim,
                                           nhead=args.nhead).cuda()
                    test_model = nn.DataParallel(test_model).cuda()
                    test_model.load_state_dict(pth)
                    print('case 1')


                # test dataloader 선언
                bs = 20
                df = pd.read_csv('./data/spotify_million_playlist/url.csv')
                test_dataset = TestJsonAudio(df=df, data_dir='./data/json_audio')
                test_loader = DataLoader(test_dataset,batch_size=bs,num_workers=2,shuffle=False, drop_last= False)
                print(f'Dataset 준비 : {len(test_dataset)}')


                # 모델에 넣어서 벡터뽑기
                pop_vector = torch.zeros((2000,2048))
                test_model.eval()
                for i in tqdm(range(args.epochs)): # epoch만큼 노래 반복으로 들려줌
                    for idx,audio in tqdm(enumerate(test_loader)):
                        audio = audio.float() # [B, 1, 48000]
                        audio = image_augmentation(audio.numpy()).cuda() # [B, 3, 256, 256]
                        pred = test_model(audio)
                        # pred = torch.squeeze(pred,dim=0)
                        pop_vector[idx*bs:idx*bs+bs] += pred.detach().cpu()/args.epochs
                        del(pred)
                vector = pop_vector.detach().cpu().clone() # [2000, 2048]

                # vector만 저장
                vector_dict = {'vector':vector.tolist()}
                with open(PATH + '/vector.json', 'w') as f:
                    json.dump(vector_dict, f)

                # Test하기
                test(vector,n_show=args.n_show)

            else:
                try:
                    try:
                        model = Siamusic(backbone=args.backbone,
                                dim=args.dim,
                                nhead=args.nhead).cuda()
                        model = nn.DataParallel(model).cuda()
                        model.load_state_dict(pth)
                        print('case 1')
                    except:
                        model = Siamusic(backbone=args.backbone,
                                dim=args.dim,
                                nhead=args.nhead).cuda()
                        model = nn.DataParallel(model).cuda()
                        model.load_state_dict(pth) 
                        print('case 2')
                except:
                    try:
                        model = Siamusic(backbone=args.backbone,
                                dim=args.dim,
                                nhead=args.nhead).cuda()
                        model.load_state_dict(pth) 
                        print('case 3')
                    except:
                        model = Siamusic(backbone=args.backbone,
                                dim=args.dim,
                                nhead=args.nhead).cuda()
                        model.load_state_dict(pth)
                        print('case 4')
                test_model = TestEncoder(backbone=args.backbone,
                                        dim=args.dim,
                                        nhead=args.nhead).cuda()
                test_model = nn.DataParallel(test_model,device_ids=[0,1]).cuda()
                model = model.cuda()
                test_model.encoder = model.encoder
                del(model)


            # test dataloader 선언
            bs = 100
            df = pd.read_csv('./data/spotify_million_playlist/url.csv')
            test_dataset = TestJsonAudio(df=df, data_dir='./data/json_audio')
            test_loader = DataLoader(test_dataset,batch_size=bs,num_workers=2,shuffle=False, drop_last= False)
            print(f'Dataset 준비 : {len(test_dataset)}')


            # 모델에 넣어서 벡터뽑기
            pop_vector = torch.zeros((2000,2048))
            test_model.eval()
            for i in tqdm(range(args.epochs)): # epoch만큼 노래 반복으로 들려줌
                for idx,audio in tqdm(enumerate(test_loader)):
                    audio = audio.float().cuda() # [B, 1, 48000]
                    pred = test_model(audio)
                    # pred = torch.squeeze(pred,dim=0)
                    pop_vector[idx*bs:idx*bs+bs] += pred.detach().cpu()/args.epochs
                    del(pred)
            vector = pop_vector.detach().cpu().clone() # [2000, 2048]

            # vector만 저장
            vector_dict = {'vector':vector.tolist()}
            with open(PATH + '/vector.json', 'w') as f:
                json.dump(vector_dict, f)

            # Test하기
            test(vector,n_show=args.n_show)

        else: # 이미 vector가 있다면
            with open(PATH + '/vector.json', 'r') as f:
                v = json.load(f)['vector']
                vector = torch.Tensor(v)
            test(vector,n_show=args.n_show)

    print("Process Complete : it took {time:.2f} minutes".format(time=(time.time()-start)/60))

if __name__ == '__main__':
    main()