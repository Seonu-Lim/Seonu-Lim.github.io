---
title: "Time Series Prediction With LSTM"
categories:
  - Python
last_modified_at: 2020-09-20T13:00:00+09:00
use_math: false
---

# 1. Intro.
이번 포스트에서 할 시계열 예측은 explanatory variable 이 있는 형태이다. 즉, input data는 대략 아래와 같은 모습일 것이다.

|date      |Target| X_1 | X_2 | ... | X_n |
|----------|------|-----|-----|-----|-----|
|2020-01-01| 1.036| 3.3 | 100 | ... | 17.3|
|       ...| ...  | ... | ... | ... | ... |

그리고 output으로는 Target 만 뱉어내도록 만들자. Pytorch의 LSTM을 사용하였고 학습을 위해 gpu 를 사용하였다.

예시를 위해 [엔씨소프트의 주가 데이터](https://kr.investing.com/equities/ncsoft-corp-historical-data)를 사용하였다. 우선 필요한 패키지들을 임포트한다.

# 2. Importing Packages

```python
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
from datetime import timedelta
import matplotlib.pyplot as plt

import plotly
import plotly.io as pio
pio.renderers.default = "notebook_connected"
```

마지막 plotly는 visualization을 위한 것이니 matplotlib로만 시각화하고자하면 굳이 import 하지 않아도 된다. 링크에서 데이터를 가져와서 간단히 preprocessing 을 하고(pandas로 읽어오면 모든 element가 string이기 때문에 약간의 preprocessing 이 필요하다.), 일일 최고가와 거래량만 사용하도록 하자. target이 일일 최고가이고, 거래량을 explanatory variable 로 사용된다. 아래의 코드로 preprocessing을 하였다.

# 3. Preprocessing

```python
df = pd.read_csv('data/ncsoft.csv')
df.columns = ['date','closing','open','high','low','volume','fluct']
df['date'] = pd.to_datetime(df.date,format='%Y년 %m월 %d일')
def strtoint(x) :
    return int(x.replace(',',''))
df.iloc[:,1:-2] = df.iloc[:,1:-2].applymap(strtoint)
def ktonum(x) :
    try :
        return float(x.replace('K','')) * 1000
    except :
        return 0
df['volume'] = df['volume'].apply(ktonum)
df = df.set_index('date')
df = df.iloc[:,[2,4]].sort_index(axis=0)
```
7일의 가격과 거래량을 보고 그 다음 1일을 맞히는 LSTM 모델을 만들 것이다. 따라서, 이에 맞게 데이터를 바꿔줘야 한다. [Time Series Forecasting with LSTMs for Daily Corona Virus Cases using Pytorch in Python](https://www.curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/) 을 참고했다. MinMax scaling을 한 뒤, X는 7일, y는 그 다음 1일로 잘라준다. 그리고 거기에 더해 10 fold cross validation을 하려고 한다. 아래의 그림과 같이 데이터를 10개로 나눈 뒤, 하나씩 뽑아서 validation set 으로 사용하는 식이다. 엔씨소프트 데이터의 경우 10 fold 를 한 경우와 하지 않은 경우에 큰 차이가 없었다. 그래도 아무튼 해보자. train 과 test 로 나눈 뒤, train 은 10개로 나눠두면 된다. `torch.split`은 나눠진 tensor들의 tuple을 return한다.

```python
train = df.iloc[:2000]
test = df.iloc[2000:]

scaler = MinMaxScaler()
scaler = scaler.fit(train)
data = scaler.transform(train)
test_data = scaler.transform(test)

seq_length = 7
y_len = 1
n_f = df.shape[1]

def create_sequences(data,seq_length,y_length) :
    xs = []
    ys = []
    for i in range(len(data)-seq_length-y_length) :
        x= data[i:(i+seq_length)]
        y= [i[0] for i in data][(i+seq_length):(i+seq_length+y_length)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
X,y = create_sequences(data,seq_length,y_len)
X_test, y_test = create_sequences(test_data,seq_length,y_len)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
X_split = torch.split(X,X.shape[0]//10 + 1)
y_split = torch.split(y,y.shape[0]//10 + 1) #10 fold
X_test = torch.from_numpy(X_test).float().to('cuda')
y_test = torch.from_numpy(y_test).float().to('cuda')
```
# 4. Model Defining

이제 LSTM이 들어간 Predictor class 를 정의해보자. `__init__` 에서 변수나 함수를 정의하고 `forward` 에서 이들을 이용하여 lstm 의 last hidden layer를 linear transformation 시켜서 하나의 normalize된 price 로 내보낸다. LSTM 에서 나오는 feature의 개수는 n_hidden으로 정할 수 있다.

```python
class Predictor(nn.Module) :
    def __init__(self, n_features, n_hidden,seq_len,y_length, n_layers=2):
        super(Predictor,self).__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers,
            dropout = 0
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=y_length)

    def forward(self, sequences):
        lstm_out, self_hidden = self.lstm(sequences)
        last_time_step = lstm_out[:, -1]      
        y_pred = self.linear(last_time_step)
        return y_pred
```
Pytorch는 진짜 최고의 라이브러리인 것 같다 코드가 이렇게나 간단하다니... Tensorflow는 사용 안해봐서 모르겠다 맨날 코드 가져와서 돌리면 버전 오류나서 포기했다. 이제 train loop를 짜면 끝이다. 10 fold validation을 하기로 했던 걸 잊지 말자. RMSE loss와 ADAM optimizer를 사용하기로 한다.

# 5. Training

```python
def train_model(model,X_split,y_split):
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100  # epochs per fold
    train_hist = np.zeros((num_epochs,(len(X_split))))
    test_hist = np.zeros((num_epochs,(len(X_split))))
    model = model.to('cuda')
    for i in range(len(X_split)) : # k fold for time series
        X_test = X_split[i].to('cuda')
        y_test = y_split[i].to('cuda')
        idx = [k for k in range(len(X_split)) if k != i]
        X_train = torch.cat([X_split[x] for x in idx])
        y_train = torch.cat([y_split[x] for x in idx])
        dataset = TensorDataset(X_train,y_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
        print(str(i+1)+'th fold training start')
        for t in range(num_epochs):
            for batch_idx, samples in enumerate(dataloader):
                X_train, y_train = samples
                X_train = X_train.to('cuda')
                y_train = y_train.to('cuda')
                y_pred = model(X_train)
                loss = torch.sqrt(loss_fn(y_pred.float(), y_train))  
                with torch.no_grad():
                    y_test_pred = model(X_test)
                    test_loss = torch.sqrt(loss_fn(y_test_pred.float(), y_test))
                test_hist[t,i] += test_loss.item()
                train_hist[t,i] += loss.item()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            train_hist[t,i] /= len(dataloader)
            test_hist[t,i] /= len(dataloader)
            print('Epoch {:3d} train loss: {:.4f} test loss: {:.4f}'.format(t + 1, train_hist[t,i], test_hist[t,i]))

    return model.eval(), train_hist, test_hist
```

삼중 loop이기 때문에 좀 복잡해 보일 수는 있지만 그냥 10fold loop 안에 epoch loop 안에 dataloader loop 일 뿐이니 안심하자. 10fold를 하지 않는다면 dataloader이 train 전에 정의되고 맨 바깥의 loop 가 없어질 것이다. 즉, 아래와 같다.

```python
dataset = TensorDataset(X_train,y_train)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)
def train_model(model,train_data,train_labels,test_data = None,test_labels=None):
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 200  
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    model = model.to('cuda')
    for t in range(num_epochs):
        for batch_idx, samples in enumerate(dataloader):
            X_train, y_train = samples
            X_train = X_train.to('cuda')
            y_train = y_train.to('cuda')
            y_pred = model(X_train)
            loss = torch.sqrt(loss_fn(y_pred.float(), y_train))  #RMSE
            if test_data is not None:
                with torch.no_grad():
                    y_test_pred = model(X_test)
                    test_loss = torch.sqrt(loss_fn(y_test_pred.float(), y_test))
                test_hist[t] += test_loss.item()
            else :
                print(f'Epoch {t} Batch {batch_idx} train loss: {loss.item()}')
            train_hist[t] += loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        train_hist[t] /= len(dataloader)
        test_hist[t] /= len(dataloader)
        print('Epoch {:3d} train loss: {:.4f} test loss: {:.4f}'.format(t + 1, train_hist[t], test_hist[t]))
    return model.eval(), train_hist, test_hist
```
이렇게 하면 epoch 당 평균 loss 를 저장하게 된다. train을 시작하면 아래와 같이 결과가 찍히도록 했다.

```python
model = Predictor(n_features=n_f, n_hidden=256, seq_len=seq_length, y_length=y_len)
model, train_hist, test_hist = train_model(model, X_split, y_split)
```
```
1th fold training start
Epoch   1 train loss: 0.2630 test loss: 0.1755
Epoch   2 train loss: 0.2192 test loss: 0.1187
Epoch   3 train loss: 0.0780 test loss: 0.0616
Epoch   4 train loss: 0.0258 test loss: 0.0235
Epoch   5 train loss: 0.0207 test loss: 0.0226
Epoch   6 train loss: 0.0196 test loss: 0.0225
Epoch   7 train loss: 0.0185 test loss: 0.0215
Epoch   8 train loss: 0.0188 test loss: 0.0215
Epoch   9 train loss: 0.0179 test loss: 0.0207
Epoch  10 train loss: 0.0176 test loss: 0.0201
...
2th fold training start
...
```
# 6. Result

loss의 변화를 그려보면 아래와 같았다.
```python
plt.plot(train_hist.flatten())
```
![train_hist](/assets/train_hist.png)
```python
plt.plot(test_hist.flatten())
```
![test_hist](/assets/test_hist.png)

training 후에는 loss 가 아주 작게 나오고 있음을 확인할 수 있었다.
이제 아직 사용하지 않은 test set 을 이용하여 prediction result 를 그려보자.

```python
with torch.no_grad() :
    preds = []
    for i in range(len(X_test)):
        testdata = X_test[i:i + 1]
        y_test_pred = model(testdata)        
        pred = y_test_pred[:, -1].cpu().numpy()
        preds.append(pred)
```
`preds`에 결괏값들을 저장하였다. 이제 이걸 MinMax Scaling 이전의 값으로 되돌리면 되는데, 문제는 train 과 preds의 모양이 다르다는 것이다. 그냥 inverse scaling 하면 모양 다르다고 오류 난다. 따라서, 억지로 train 과 모양을 맞춰준 다음에 inverse scaling 하고 맨 앞의 elements들만 가져오면 된다.

```python
y_t = y_test[:,-1].cpu().numpy()
y_t = [np.repeat(t,n_f) for t in y_t]
inv_y = scaler.inverse_transform(y_t)
true_y = [i[0] for i in inv_y]
temp = [np.repeat(p,n_f) for p in preds]
inv_tr = scaler.inverse_transform(temp)
true_pr = [i[0] for i in inv_tr]
true_y = pd.DataFrame(true_y)
true_pr = pd.DataFrame(true_pr)
vis_df = pd.concat([true_y,true_pr],axis=1)
vis_df.columns = ['real','predict']
vis_df = vis_df.set_index(test.index[(seq_length+y_len):(len(true_y)+seq_length+y_len)])
fig = vis_df.iplot(asFigure=True,kind = 'scatter')
fig.show()
```
나는 원소를 반복시켜서 모양을 맞췄지만 굳이 그러지 않고 0을 붙여줘도 된다. 아무튼 결과적으로는 이런 그래프가 나온다. plotly 로 interactive plot을 만들어보았는데, 확대를 하지 않고 보면 꽤 정확해 보인다.
{% include plot.html %}

하지만 확대를 해보면, 실제 데이터가 정확히 1일씩 밀려있는 게 prediction 이라고 말해도 무방하다. 즉, 관측한 7일 중 마지막 날과 예측치가 비슷할 것이라고 생각한 것이다. 아무리 error이 작게 나와도 이런 식이면 예측에 아무런 의미가 없다! error이 좀 크더라도 증감을 맞히고, 밀려있는 현상을 완화시킬 수는 없을까? 여러가지 시도를 해보고 있지만 쉽지 않은 것 같다. Time Series Prediction이라고 검색했을 때 나오는 결과들 중 대다수가 이렇게 밀려있는 것을 생각하면... 추후에는 이 부분을 개선하려고 해봤던 시도들에 대해서 작성해보도록 하겠다.
