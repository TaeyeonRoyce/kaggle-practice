# Import Module
import pandas as pd
import numpy as np
import os
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization)
from tensorflow import keras
from copy import copy


###################################### Function ######################################
# 필요한 Data를 모두 가져온다.
def import_Data(file_path):
    result = dict()
    for file in os.listdir(file_path):
        file_name = file[:-4]
        result[file_name] = pd.read_csv(file_path + "/" + file)

    return result


# Rawdata 생성
def make_Rawdata(dict_data):
    dict_key = list(dict_data.keys())
    test_Dataset = pd.merge(dict_data["gender_submission"], dict_data["test"], how='outer', on="PassengerId")
    Rawdata = pd.concat([dict_data["train"], test_Dataset])
    Rawdata.reset_index(drop=True, inplace=True)

    return Rawdata


# 불필요한 컬럼 제거
def remove_columns(DF, remove_list):
    # 원본 정보 유지를 위해 copy하여, 원본 Data와의 종속성을 끊었다.
    result = copy(Rawdata)

    # PassengerId를 Index로 하자.
    result.set_index("PassengerId", inplace=True)

    # 불필요한 column 제거
    for column in remove_list:
        del (result[column])

    return result


# 결측값 처리
def missing_value(DF, key=None):
    # Cabin 변수를 제거하자
    del (DF["Cabin"])

    if key == "mean":
        DF["Age"] = DF["Age"].fillna(np.mean(DF["Age"]))

    elif key == "median":
        DF["Age"] = DF["Age"].fillna(np.median((DF["Age"].dropna())))

    # 결측값이 있는 모든 행은 제거한다.
    DF.dropna(inplace=True)


# 원-핫 벡터
def one_hot_Encoding(data, column):
    # 한 변수 내 빈도
    freq = data[column].value_counts()

    # 빈도가 큰 순서로 용어 사전 생성
    vocabulary = freq.sort_values(ascending=False).index

    # DataFrame에 용어 사전 크기의 column 생성
    for word in vocabulary:
        new_column = column + "_" + str(word)
        data[new_column] = 0

    # 생성된 column에 해당하는 row에 1을 넣음
    for word in vocabulary:
        target_index = data[data[column] == word].index
        new_column = column + "_" + str(word)
        data.loc[target_index, new_column] = 1

    # 기존 컬럼 제거
    del (data[column])


# 스케일 조정
def scale_adjust(X_test, X_train, C_number, key="min_max"):
    if key == "min_max":

        min_key = np.min(X_train[:, C_number])
        max_key = np.max(X_train[:, C_number])

        X_train[:, C_number] = (X_train[:, C_number] - min_key) / (max_key - min_key)
        X_test[:, C_number] = (X_test[:, C_number] - min_key) / (max_key - min_key)

    elif key == "norm":

        mean_key = np.mean(X_train[:, C_number])
        std_key = np.std(X_train[:, C_number])

        X_train[:, C_number] = (X_train[:, C_number] - mean_key) / std_key
        X_test[:, C_number] = (X_test[:, C_number] - mean_key) / std_key

    return X_test, X_train


######################################################################################


################################## Global Variable ###################################
file_path = "./datas"
remove_list = ["Name", "Ticket"]
######################################################################################
# Data Handling
# 0. Rawdata 생성
Rawdata_dict = import_Data(file_path)
Rawdata = make_Rawdata(Rawdata_dict)

# 1. 필요 없는 column 제거
DF_Hand = remove_columns(Rawdata, remove_list)

# 2. 결측값 처리
missing_value(DF_Hand)

# 3. One-Hot encoding
one_hot_Encoding(DF_Hand, 'Pclass')
one_hot_Encoding(DF_Hand, 'Sex')
one_hot_Encoding(DF_Hand, 'Embarked')

# 4. 데이터 쪼개기
# Label 생성
y_test, y_train = DF_Hand["Survived"][:300].to_numpy(), DF_Hand["Survived"][300:].to_numpy()

# 5. Dataset 생성
del (DF_Hand["Survived"])
X_test, X_train = DF_Hand[:300].values, DF_Hand[300:].values

# 6. 특성 스케일 조정
X_test, X_train = scale_adjust(X_test, X_train, 0, key="min_max")
X_test, X_train = scale_adjust(X_test, X_train, 1, key="min_max")
X_test, X_train = scale_adjust(X_test, X_train, 2, key="min_max")
X_test, X_train = scale_adjust(X_test, X_train, 3, key="min_max")
######################################################################################


######################################## Model #######################################
# 모델 생성
model = keras.Sequential()
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(16, activation='relu'))
# 마지막 Dropout은 좀 크게 주자
model.add(Dropout(0.50))
model.add(Dense(1, activation='sigmoid'))

# 모델 Compile
opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=opt,
              loss="binary_crossentropy",
              metrics=["binary_accuracy"])
######################################################################################


model.fit(X_train, y_train, epochs = 100)
pred = model.predict(X_test).reshape(X_test.shape[0])
pred = np.where(pred > 0.5, 1, 0)
accuracy = 1 - (np.where((pred - y_test) == 0, 0, 1).sum()/len(y_test))
print("Accuracy:", accuracy)