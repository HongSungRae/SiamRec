# library
import torch
import pandas as pd
import torch.nn.functional as F
import os



def get_reward(show,candidates):
    '''
    show 기준으로 candidates들이 얼마나 가까운지 구합니다.
    Rank를 기준으로 줄세웁니다.
    이후 reward 공식에 의해 reward를 반환합니다.
    '''
    n_show = int(show.shape[0]/100) # 15
    n_candidates = int(candidates.shape[0]/100) # 5
    reward = 0
    for i in range(100):
        show_vector = torch.sum(show[i*n_show:i*n_show+n_show],dim=0)/n_show
        sim = torch.zeros(n_candidates*100)
        for idx in range(n_candidates*100):
            cos_sim = F.cosine_similarity(show_vector,candidates[idx],dim=0)
            sim[idx] = cos_sim
        ranks = torch.argsort(sim)
        rank_focus = ranks.tolist()[i*n_candidates:i*n_candidates+n_candidates]
        for rank in rank_focus:
            reward += 1/(rank//n_candidates+1) * (1/n_candidates)
    return reward


if __name__ == '__main__':
    show = torch.randn((1500,2048))
    candidates = torch.randn((500,2048))
    reward = get_reward(show,candidates)
    print(reward)