import torch
import pandas as pd



class PlaylistReward(object):
    def __init__(self,model,epoch,df,n_playlist,n_show,dim=2048,shuffle=False):
        self.model = model
        self.epoch = epoch
        self.df = df
        self.n_playlist = n_playlist
        self.n_show = n_show
        self.n_candidate = n_playlist - n_show
        self.dim = dim
        self.shuffle = shuffle
        self.candidate = torch.zeros((n_playlist*self.n_candidate,dim))
        self.song_info = {}
        if shuffle: # df를 20개 단위로 셔플함
            pass
        # ====== #

        self.df_show = pd.DataFrame({'pid', 'artist', 'track', 'idx'})
        self.df_candidate = pd.DataFrame({'pid', 'artist', 'track', 'idx'})

        for i in range(n_playlist):
            
        # 5곡 x 100 곡을 가져와서
        # song_info에 몇번 인덱스가 누구의 무슨 노래인지 저장함
        # 모델에 epoch만큼 포워드하여 평균치를 self.candidate에 저장함


    def run(self):
        reward = 0
        pid = self.df['pid'][0]
        # pid에 대해서 돌면서
        # 15곡을 가져오고
        # 모델에 epoch만큼 포워드해서 평균치를 구함
        # 이 평균치를 self.candidate와 비교해 GT는 몇번째로 가까운지 찾음
        # 총 5개의 가까움(=index)를 
        # self.get_reward_score에 넣어서 reward점수를 뽑는다
        # reward에 누적합
        score = self.get_reward_score(index)
        reward += score
        
        return reward

    def get_reward_score(self,*idx):
        score = 0
        for rank in idx:
            score += (rank//self.n_candidate+1)
        score /= self.n_candidate
        return score