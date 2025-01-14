## machine learning implementation
기계학습 알고리즘 직접 구현하는 프로젝트

## 선형회귀 알고리즘
### 사용 라이브러리
- numpy
- pandas

### 데이터셋
- 집값 데이터

### 모델 구조
선형 회귀 모델은 다음과 같은 방정식을 따릅니다:
y = XW

학습 과정 (경사하강법)
손실 함수(평균 제곱 오차, MSE)를 최소화하기 위해 경사하강법을 사용하여 가중치를 업데이트합니다.

성능 평가 (R² 점수)
모델의 성능은 결정 계수를 통해 평가됩니다.

## 최근접이웃 알고리즘
### 모델 구조
학습 - 메모리에 학습 데이터셋의 feature matrix, class vector 저장
예측 - 저장된 feature matirx와 L1 distance를 구한 후 가장 가까운 샘플의 클래스로 예측

## VAE
### 모델구조
encoder - 입력 데이터를 latent space로 매핑하여 평균과 로그 분산 반환  
decoder - 평균과 로그 분산을 활용하여 입력 데이터를 복원  

train - 학습에 관한 코드  
eval - 테스트 데이터에 시각화 및 원본과 비교

### 결과
![image](https://github.com/user-attachments/assets/893299c6-e1dc-4f39-87c1-0f909c4bbfa0)

## GAN
### 모델구조
generator - 노이즈로 부터 이미지 생성  
discriminator - 진짜 이미지와, 생성된 이미지를 구분  

loss function - cross entropy 이용  

### 결과
![image](https://github.com/user-attachments/assets/948a25a3-5b9a-4871-a7ec-9ba969c58f5a)
  
(학습 시간이 오래 걸려, 임시로 에포크 1로 진행한 결과)
