# library
from numpy.random import choice
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

# local
from utils import *
from dataset import JsonAudio, TestJsonAudio
from loss import SiamusicLoss
from augmentation import sungrae_pedal, random_mix
from simsiam import Siamusic, TestEncoder
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
                    help='train epoch',choices=['pedalboard','randommix','iamge'])
parser.add_argument('--patchs', default=12, type=int,
                     help='ramdom mix augmentation patchs')
parser.add_argument('--from_scratch', default=False, type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='How To Make TRUE? : --from_scratch, Flase : --no-from_scratch')
parser.add_argument('--threshold', default=0.5, type=float,
                    help='MTA 에서 confidence가 얼마 이상이면 1로 예측했다고 할 것인가?')

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
            sys.exit('곧 업데이트 예정')
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
def test(model, test_loader, test_logger, args=None, n_test=10):
    print("=================== Test Start ====================")
    model.eval()
    reward = AverageMeter()

    df = pd.read_csv('./data/spotify_million_playlist/url.csv')
    df_show = pd.DataFrame(df.columns) # 15곡 split
    df_candidates = pd.DataFrame(df.columns) # 5곡 split
    for tst in range(n_test): # 10회 반복실험
        # 평균 Tensor
        show = torch.zeros((1500,args.dim))
        candidates = torch.zeros((500,args.dim))

        # df를 20개 단위로 shuffle
        temp_df = df.copy()
        for i in range(100):
            df[i*20:i*20+20] = temp_df[i*20:i*20+20].sample(frac=1)
            df_show = df_show.append(df[i*20:i*20+15],ignore_index=True)
            df_candidates = df_candidates.append(df[i*20+15:i*20+20],ignore_index=True)
        df = df.reset_index(drop=True)
        df_show = df_show.reset_index(drop=True)
        df_candidates = df_candidates.reset_index(drop=True)

        show_dataset = TestJsonAudio(df_show)
        candidates_dataset = TestJsonAudio(df_candidates)
        show_loader = DataLoader(show_dataset,batch_size=15,num_workers=2)
        candidates_loader = DataLoader(candidates_dataset,batch_size=5, num_workers=2)

        for _ in range(30):
            for i,audio in enumerate(show_loader):
                audio = audio.cuda()
                y = model(audio)
                y = y.detach()
                show[i*15:i*15+15] += y

            for i,audio in enumerate(candidates_loader):
                audio = audio.cuda()
                y = model(audio)
                y = y.detach()
                show[i*5:i*5+5] += y
        
        show = show/30
        candidates = candidates/30

        r = get_reward(show,candidates)
        reward.update(r)

    print(f'Reward Score : {reward.avg}±{reward.std}')
    print("=================== Test End ====================")




def main():
    # define environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    model = Siamusic(backbone=args.backbone,
                     dim=args.dim,
                     nhead=args.nhead).cuda()

    
    # pre-training or fine-tuning
    if args.from_scratch: ## pre-training
        print('스크래치부터 학습되고 Test까지 진행합니다.')

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
        print('학습된 Siam 모델을 불러와서 Test를 진행합니다.')
        save_path=args.save_path+args.backbone+'_'+args.augmentation+'_'+args.optim+'_'+args.fma
        
        # 모델 불러오기 & pretrain모델 주입
        '''
        https://justkode.kr/deep-learning/pytorch-save
        https://tutorials.pytorch.kr/beginner/saving_loading_models.html
        '''

        PATH = './exp_' + args.backbone + '_' + args.augmentation + '_' + args.optim+'_'+args.fma
        pth_file = args.backbone+'_'+args.augmentation+'_100.pth'
        try:
            model.load_state_dict(torch.load(PATH+'/'+pth_file))
            test_model = TestEncoder(backbone=args.backbone,
                                     dim=args.dim,
                                     nhead=args.nhead).cuda()
            test_model.encoder = model.encoder
        except:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            test_model = TestEncoder(backbone=args.backbone,
                                     dim=args.dim,
                                     nhead=args.nhead).cuda()
            model = Siamusic(backbone=args.backbone,
                             dim=args.dim,
                             nhead=args.nhead).cuda()
            model.load_state_dict(torch.load(PATH+'/'+pth_file))
            test_model.encoder = model.encoder


        # test dataloader 선언
        test_dataset = JsonAudio(data_dir='./data/json_audio')
        test_loader = DataLoader(test_dataset,batch_size=20,num_workers=2)

        # logger
        test_logger = Logger(os.path.join(save_path, 'train_loss.log'))

        # 테스트시작
        test(test_model, test_loader, test_logger, args)

    print("Process Complete : it took {time:.2f} minutes".format(time=(time.time()-start)/60))

if __name__ == '__main__':
    main()