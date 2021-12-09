from __future__ import unicode_literals

from selenium import webdriver
from bs4 import BeautifulSoup as bs
import pandas as pd
from selenium.webdriver.common.keys import Keys
import time
import youtube_dl
import pandas as pd
from pytube import YouTube
from tqdm import tqdm
import os


def get_YouTube_url(df):
    driver = webdriver.Chrome('./chromedriver.exe')

    # df2 = df['url'].copy()
    # for idx, item in df.iterrows():
    #     pid,artist_name,track_name,_ = item
    #     keyword = artist_name + ' ' + track_name
    #     search_url = 'https://www.youtube.com/results?search_query={}'.format(keyword)

    #     driver.get(search_url)
    #     soup = bs(driver.page_source, 'html.parser')

    #     video_url = soup.select('a#video-title')
        
    #     for i in video_url:
    #         df2[idx] = '{}{}'.format('https://www.youtube.com',i.get('href'))
    #         break

    for i in tqdm(range(len(df))):
        artist = df['artist_name'][i]
        track = df['track_name'][i]
        keyword = artist + ' ' + track
        search_url = 'https://www.youtube.com/results?search_query={}'.format(keyword)
        driver.get(search_url)
        soup = bs(driver.page_source, 'html.parser')
        video_url = soup.select('a#video-title')
        for j in video_url:
            title = j.text.strip()
            if isEnglishOrKorean(title)=='k':
                continue
            else:
                df['url'][i] = '{}{}'.format('https://www.youtube.com',j.get('href'))
                break

    # df['url'] = df2
    df.to_csv('./data/spotify_million_playlist/url.csv',index=False)
    # driver.close()
    print('Url 생성 완료...')



def download_mp3(artist=None,title=None,link=None):
    assert link!=None and artist!=None and title!=None
    file = artist + '-' + title + '.mp3'
    ydl_opts = {
        'outtmpl':'./data/mp3/'+ file,
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])


# https://pytube.io/en/latest/api.html#pytube.Stream.download
# 이게 훨씬 빠름
def download_mp4(artist=None,title=None,link=None):
    assert link!=None and artist!=None and title!=None
    down, dupl, error = 0,0,0
    error_term = ['/','\"','<','>','\\','|',':','*','?'] # 얘네는 윈도우에서 제목에 못 있는다
    local = os.listdir('./data/audio')
    try:
        file = artist + '-' + title + '.mp4'
        if file in local:
            dupl = 1
            print(f'Alert : {artist}-{title} already exists.')
        else:
            yt = YouTube(link) 
            stream = yt.streams.filter(only_audio=True).first()
            for item in error_term:
                if item in file:
                    file = file.replace(item,'^')
            stream.download('./data/audio',file)
            down = 1
    except:
        print(f'Caution : {artist}-{title} is invalild.')
        error = 1
    return down, dupl, error


# 광고가 뜨는데 광고는 모두 한국어다
# 광고 거르기용
def isEnglishOrKorean(input_s):
    k_count = 0
    e_count = 0
    for c in input_s:
        if ord('가') <= ord(c) <= ord('힣'):
            k_count+=1
        elif ord('a') <= ord(c.lower()) <= ord('z'):
            e_count+=1
    return "k" if k_count>1 else "e"



# 2000개의 곡이 audio 폴터에 모두 있는지 확인하는 함수
def checker():
    c = 0
    df = pd.read_csv('./data/spotify_million_playlist/url.csv')
    musics = os.listdir('./data/audio')
    error_term = ['/','\"','<','>','\\','|',':','*','?'] # 얘네는 윈도우에서 제목에 못 있는다
    for i,item in enumerate(df.iterrows()):
        _,artist,track,url = item[1]
        music = artist + '-' + track + '.mp4'
        for item in error_term:
            if item in music:
                music = music.replace(item,'^')
        if music not in musics:
            c += 1
            print(f'Caution : {artist}-{track} not in AUDIO')
    print(f'총 {c}곡이 없습니다.')



# 이 main함수는 ./data/spotify_million_playlist/follower상위100.csv에 있는 playlist에서
# 각각의 playlist 중 앞 20곡을 test_audio에 다운로드합니다.
def main():
    start = time.time()
    df = pd.read_csv('./data/spotify_million_playlist/url.csv')
    
    print('== Now on get YouTube links... ==')
    # get_YouTube_url(df)
    print('== Now on starts crawling... ==')

    total = [0,0,0]
    for i,item in enumerate(df.iterrows()):
        pid,artist,track,url = item[1]
        down, dupl, error = download_mp4(artist,track,url)
        total[0] += down
        total[1] += dupl
        total[2] += error
    print(f'== Download Completed | Down : {total[0]} | Dupl : {total[1]} | Error : {total[2]} | Sum : {sum(total)} ==')
    print(f'It took {(time.time()-start)/60:.2f} minutes')



if __name__ == '__main__':
    ## Test downloader
    # test_link = "https://www.youtube.com/watch?v=74_yqNBhQbA" # Yerin Baek - Antifreeze
    # download_mp3('Yerin Baek','Antifreeze',test_link)
    # download_mp4('Yerin Baek','Antifreeze',test_link)

    ## Test url crawler
    # df = pd.read_csv('./data/spotify_million_playlist/url.csv')
    # get_YouTube_url(df[0:2])
    
    
    # main()
    checker()