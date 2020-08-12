# logistic_mf

### 파일 설명 

* data/EDA.ipynb : 데이터 EDA 
* Loader.py      : 데이터 로드 및 전처리 클래스 
* Metric.py      : top@K 메트릭 계산 클래스
* model/GMF.py   : generalized matrix factorization 모델 
* model/MLP.py   : multi-layer perceptron 모델 
* model/NeuMF.py : neural collaborative filtering 모델


### 데이터

* usersha1-artmbid-artname-plays.tsv : lastfm의 음악 스트리밍 데이터로 user, item(artist), plays(play횟수)로 이뤄짐. 
* 출처: http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html


### 코드 실행 예시
```
python Run.py
```


### reference
* https://arxiv.org/pdf/1708.05031.pdf?source=post_page
* https://medium.com/@victorkohler/collaborative-filtering-using-deep-neural-networks-in-tensorflow-96e5d41a39a1



### update

* Last Update Date: 2020/8/12
