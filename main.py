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

# local
from utils import *
from dataset import JsonAudio
from loss import SiamusicLoss
from augmentation import sungrae_pedal, random_mix
from simsiam import Siamusic


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
def test(model, test_loader, test_logger):
    print("=================== Test Start ====================")
    model.eval()
    path_candidate = './data/candidate'
    path_show = './data/show'

    # candidate를 모델에 여러번 넣어서 평균 vector뽑기
    candidates = torch.zeros((5*100,args.dim))
    for song in next(os.walk(path_candidate))[2]:
        for iter in range(30):
            with open(path_candidate+'/'+song, 'r') as f:
                waveform = np.array(json.load(f)['audio'],dtype=float)
            random_idx = np.random.randint(low=0, high=int(waveform.shape[-1] - 48000))
            waveform = waveform[0][random_idx:random_idx+48000]
            audio = np.expand_dims(waveform, axis = 0) # expand to [1,48000]
            audio = torch.Tensor(audio)
            audio = audio.float().cuda()
            vector = model(audio)
            candidates[iter] += vector
    candidates = candidates/30

    # playlist별로 순회하면서 15곡의 평균 vector 뽑기
    playlists = torch.zeros((100,args.dim))

    # playlists를 순회하면서 candidates를 sort하고 REWARD 점수 뽑아 모두 합산하기
    # 
    print("=================== Test End ====================")




def main():
    # define environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    model = Siamusic(backbone=args.backbone,
                     dim=args.dim,
                     nhead=args.nhead).cuda()
    model = nn.DataParallel(model)

    
    # pre-training or fine-tuning
    if args.from_scratch: ## pre-training
        print('스크래치부터 학습됩니다.')

        # save path
        save_path=args.save_path+'_'+args.backbone+'_'+args.augmentation+'_'+args.optim
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # dataset loading
        train_dataset = JsonAudio('D:/SiamRec/data/json_audio',args.input_length)
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True,drop_last=True)
        print('=== DataLoader R.e.a.d.y ===')

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
    
    else: ## fine-tuning
        print('학습된 Siam 모델을 불러와서 Test를 진행합니다.')
        # save path
        save_path=args.save_path+'_TEST_'+args.backbone+'_'+args.augmentation+'_'+args.optim
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 모델 불러오기 & pretrain모델 주입
        '''
        https://justkode.kr/deep-learning/pytorch-save
        https://tutorials.pytorch.kr/beginner/saving_loading_models.html
        '''
        PATH = './exp_' + args.backbone + '_' + args.augmentation + '_' + args.optim
        pth_file = args.backbone+'_'+args.augmentation+'_100.pth'
        model.load_state_dict(torch.load(PATH+'/'+pth_file))
        
        # test dataloader 선언
        test_loader = JsonAudio('Some Path')

        # logger
        test_logger = Logger(os.path.join(save_path, 'train_loss.log'))

        # 테스트시작
        test(model, test_loader, test_logger)

    print("Process Complete : it took {time:.2f} minutes".format(time=(time.time()-start)/60))

if __name__ == '__main__':
    main()