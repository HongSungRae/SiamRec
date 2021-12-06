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


def get_YouTube_url(df):
    #pd.set_option('mode.chained_assignment',  None)

    driver = webdriver.Chrome('./chromedriver.exe')
    df2 = df['url'].copy()

    for idx, item in df.iterrows():
        pid,artist_name,track_name,_ = item
        keyword = artist_name + ' ' + track_name
        search_url = 'https://www.youtube.com/results?search_query={}'.format(keyword)

        driver.get(search_url)
        soup = bs(driver.page_source, 'html.parser')

        video_url = soup.select('a#video-title')
        
        for i in video_url:
            df2[idx] = '{}{}'.format('https://www.youtube.com',i.get('href'))
            break

    df['url'] = df2
    df.to_csv('./data/spotify_million_playlist/url.csv',index=False)
    driver.close()
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
    try:
        yt = YouTube(link) 
        stream = yt.streams.filter(only_audio=True).first()
        file = artist + '-' + title + '.mp4'
        stream.download('./data/audio',file)
    except:
        print(f'Caution : {artist}-{title} is invalild.')



# 이 main함수는 ./data/spotify_million_playlist/follower상위100.csv에 있는 playlist에서
# 각각의 playlist 중 앞 20곡을 test_audio에 다운로드합니다.
def main():
    df = pd.read_csv('./data/spotify_million_playlist/url.csv')
    # get_YouTube_url(df)
    for i,item in enumerate(df.iterrows()):
        pid,artist,track,url = item[1]
        download_mp4(artist,track,url)



if __name__ == '__main__':
    ## Test downloader
    # test_link = "https://www.youtube.com/watch?v=74_yqNBhQbA" # Yerin Baek - Antifreeze
    # download_mp3('Yerin Baek','Antifreeze',test_link)
    # download_mp4('Yerin Baek','Antifreeze',test_link)

    ## Test url crawler
    # df = pd.read_csv('./data/spotify_million_playlist/url.csv')
    # get_YouTube_url(df[0:2])
    main()