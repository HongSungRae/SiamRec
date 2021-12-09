# library
import torch
import pandas as pd
import torch.nn.functional as F
import os



def get_reward(show,candidates):
    '''
    show 기준으로 candidates들이 얼마나 가까운지 구합니다
    이후 reward 공식에 의해 reward를 반환합니다
    '''
    # show : [1500,dim]
    # candidates : [500,dim]
    reward = 0
    for i in range(100):
        show_vector = torch.sum(show[i*15:i*15+15],dim=0)/15
        sim = torch.zeros(500)
        for idx in range(500):
            cos_sim = F.cosine_similarity(show_vector,candidates[idx],dim=0)
            sim[idx] = cos_sim
        ranks = torch.argsort(sim)
        rank_focus = ranks.tolist()[i*5:i*5+5]
        for rank in rank_focus:
            reward += 1/(rank//5+1) * 0.2
    return reward


if __name__ == '__main__':
    show = torch.randn((1500,2048))
    candidates = torch.randn((500,2048))
    reward = get_reward(show,candidates)
    print(reward)