import tensorflow as tf

import numpy as np
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import SGD

print(tf.__version__)
#%%
# [1] 데이터셋 생성

x_data = np.array([1, 2, 3, 4, 5, 6])
t_data = np.array([3, 4, 5, 6, 7, 8])
#%%
# [2] 모델 (model) 구축

model = Sequential()    # 모델

model.add(Flatten(input_shape=(1,)))       # 입력층

model.add(Dense(1, activation='linear'))   # 출력층

# model.add(Dense(1, input_shape=(1,), activation='linear'))  # 입력층 + 출력층
#%%
# [3] 모델 (model) 컴파일 및 summary

model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse')

model.summary()
#%%
# [4] 모델 학습

hist = model.fit(x_data, t_data, epochs=1000)
#%%
# [5] 모델 (model) 사용

result = model.predict(np.array([-3.1, 3.0, 3.5, 15.0, 20.1]))

print(result)