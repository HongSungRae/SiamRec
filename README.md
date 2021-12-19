# SiamRec
- KAIST KSE801 2021 Fall Final Project
- LaTeX formed report is available : [LINK](https://github.com/HongSungRae/SiamRec/blob/main/images/KSE801_20214645%ED%99%8D%EC%84%B1%EB%9E%98.pdf)

# 1. 소개
## 1.1 개요

It is a very important task to recommend items that suit the tastes of users on the music streaming platform. Unlike platforms in other domains, users continuously consume items while staying on the music platform. Until now, music recommendations have been made using the Collaborative Filtering (CF) algorithm. However, in order to properly connect new songs with users added every day, any methodolgies to solve the cold-start problem were needed. In this study, we propose a playlist recommendation using audio presentation, which has not been frequently used. A network that has trained with large amounts of audio with self-supervised learning recommends new items to the appropriate playlist based on the presentation of existing playlist songs. To train the network, we propose three novel augmentation which is applicable to audio data. Our experimental results are presented qualitatively or quantitatively.

## 1.2 성능
### 1.2.1 정량평가
![](https://github.com/HongSungRae/SiamRec/blob/main/images/performance.jpg?raw=true)
### 1.2.2 정성평가
- 1사분면에서 4사분면 방향으로 [ResNet101, ResNet50, ResNet152, Transformer] 입니다
- Axis의 수치에는 별다른 의미 없습니다
#### A. 아티스트
##### Pedalboard Augmentation
![](https://github.com/HongSungRae/SiamRec/blob/main/images/artist_pedal.png?raw=true)

##### Random Mix Augmentation
![](https://github.com/HongSungRae/SiamRec/blob/main/images/artist_random.png?raw=true)

#### B. 아티스트와 노래
##### Pedalboard Augmentation
![](https://github.com/HongSungRae/SiamRec/blob/main/images/song_pedal.png?raw=true)

##### Random Mix Augmentation
![](https://github.com/HongSungRae/SiamRec/blob/main/images/song_random.png?raw=true)

## 1.3 Collapsing?
### 1.3.1 Pedalboard Augmentation
![](https://github.com/HongSungRae/SiamRec/blob/main/images/FMA_pedalboard.png?raw=true)
### 1.3.2 Random Mix Augmentation
![](https://github.com/HongSungRae/SiamRec/blob/main/images/FMA_randommix.png?raw=true)
### 1.3.3 Spectrogram Image Augmentation
![](https://github.com/HongSungRae/SiamRec/blob/main/images/FMA_Image.png?raw=true)
---
# 2. How To Use
- 이 코드를 사용하는 방법을 설명합니다
- 2.1부터 챕터 2의 마지막까지 모두 순서대로 진행해주세요
## 2.1 환경 설정하기
1. 여러분의 환경에 git clone합니다 : ```git clone <this repo>```
2. [ffmpeg.exe](https://www.ffmpeg.org/download.html)를 디렉토리('./SiamRec)에 다운로드 받습니다
3. ```ffplay.exe```와 ```ffprobe.exe```가 함께 다운됩니다. 그렇지 않다면 구글링하여 따로 다운받아 디렉토리에 넣습니다
4. 크롤링을 위한 라이브러리와 보조 프로그램의 이유때문에 Window환경에서만 구동됩니다
5. 아래와같이 requirements package를 다운로드합니다
```
>pip install -r requirments.txt
```
## 2.2 데이터 다운로드받기
### FMA Dataset
1. pre-training에 사용된 FMA-small 데이터를 [링크](https://github.com/mdeff/fma)에서 다운로드 받으세요
2. 압축을 풀고 './SiamRec/data'에 압축을 풉니다
### Spotify Million Playlist
1. 구글링해서 받으세요
2. './SiamRec/data'에 넣습니다
### POP Music Youtube에서 크롤링하기
1. ```statistics.ipynb``` 을 큰 챕터 5번까지만 순서대로 모두 구동합니다
2. './SiamRec/data/spotify_million_playlist'에 ```url.csv```와 ```follower상위100.csv```가 생성됩니다
3. 'data/audio' 경로와 'data/audio_json'경로 폴더를 만들어줍니다
4. ```crawler.py```를 구동합니다. 링크 크롤링부터 다운로드까지 자동으로 됩니다.
5. 노래가 모두 잘 다운되었는지 확인하고싶으면 ```crawler.py```의 ```checker()```함수만 실행해보세요
6. 다운되지 않는 7곡은 수동으로 받아 넣었습니다
7. ```utils.py```에 ```mp3_to_json()```의 설명을 읽고 자신의 경로에 맞게 방금 받은 데이터를 모두 json으로 변환하여 'data/audio_json'에 저장합니다

## 2.3 학습하기 + 추론하기
```
python main.py --backbone resnet50
               --dim 2048
               --input_length 48000
               --augmentation pedalboard
               --fma small
               --batch_size 64
               --optim adam
               --epochs 100
               --from_scratch
               --n_show 15
               --comment COMMENT WHATEVER YOU WANT
               --gpu_id 1
```
- ```main.py```의 argparser를 참고하여 backbone, dim(출력 벡터의 차원), optimizer, augmentation 등을 바꿔가며 실험합니다
- nheads는 transformer를 위한 옵션입니다
- 학습이 되면 './SiamRec'경로에 모델이 저장됩니다. --no-from_scratch 옵션을 주고 --n_show를 10부터 18까지 바꿔가면서 성능을 비교해보세요

## 2.4 tSNE로 유사도 확인하기
```
python main.py --save_path ./exp_resnet50_pedalboard_adam_small
               --backbone resnet50
               --dim 2048
               --iter 30
```
---
# 감사
- 첫 플젝 잘 끝내서 모두에게 감사합니다 ;)
- ♥메리 크리스마스♥

---
# 라이센스
- MIT license를 따릅니다.
- 다음의 출처를 남긴 자유로운 사용이 가능합니다.
```
SiamRec2021, SungRae Hong, KAIST KSE, Daejeon, South Korea.
```