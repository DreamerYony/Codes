## 🍳 Humor generation / Humor converter (2023-1)
### 🍳 T5와 Few shot learning으로 유머 변환기 만들기
#### 🍳 데이터셋 : Movie Title Puns by Hämäläinen and Alnajjar(2019).
(Mika Hämäläinen and Khalid Alnajjar. 2019. 
Modelling the Socialization of Creative Agents in a Master-Apprentice Setting: The Case of Movie Title Puns.  
https://www.kaggle.com/datasets/mikahama/movie-title-puns)
#### 🍳 내용   
1. 튜플 형태로 '본래 영화 제목 - 유머러스하게 바뀐 영화 제목'을 퓨샷(n=16,32,64)으로 입력  
2. 말장난의 유형(Types of Puns)   
(1) 단어 교체(Word Replacement)  
(2) 유사 발음(Similar Sound) 이용  
(3) 유사 음절(Similar Syllable) 이용  
### 🍳 포함된 파일
#### T5_preprocessing.ipynb
- 유머를 말장난의 유형에 따라 분류하기 위한 코드
- 각 유형의 '본래 영화 제목 - 유머러스하게 바뀐 영화 제목'을 선정
- 이렇게 선정된 예시를 '좋은' 훈련 샘플로 간주
#### Few_shot_learning_with_T5.ipynb
- 선정된 퓨샷 예시를 t5모델에 넣고 훈련
- 결과 도출
### 🍳 유머 평가
- 이 프로젝트 참여자인 2인이 각각 유머 점수를 산정
- 본인(나)을 포함한 동료 1명
