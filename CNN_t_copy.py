import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint,EarlyStopping
np.random.seed(7)

# Image size set
img_rows = 28
img_cols = 28

# MNIST DataSet Load
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# 팀 데이터셋을 train에 concat
import pickle
with open("C:/Users/sb090/DeepLearning/teamdataset/all_shuffle.pkl", 'rb') as f:
    xt_train, yt_train = pickle.load(f)

x_train = np.concatenate(([x_train, xt_train]), axis=0)
y_train = np.concatenate(([y_train, yt_train]), axis=0)

# train dateset shuffle
idx = np.arange(len(x_train))
np.random.shuffle(idx)

x_train = x_train[idx]
y_train = y_train[idx]



# test셋을 mnist 데이터셋 빼고 교수님 데이터셋으로 교체
with open("C:/Users/sb090/DeepLearning/testdata2D.pkl", 'rb') as f:
    datasetL = pickle.load(f)
x_test, y_test = datasetL



# reshape
input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# normalization
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)          #x_train shape: (64000쯤...?, 28, 28, 1)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



batch_size = 128       #32~128 적절
num_classes = 10
epochs = 8


# 학습데이터에 대한 답(0 ~ 9)을 one-hot-encoding 으로 변환한다
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



# Model을 생성
model = Sequential()

# Layer 1 Convolution : 이미지 특징 검출
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# MaxPooling : 정해진 구간에서 가장 큰값만 남기고 버림
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 2
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 학습데이터 0.25% 를 과적합방지를 위해 손실 처리
model.add(Dropout(0.25))
# Flatten 2차원 데이터를 1차원으로 변환 처리
model.add(Flatten())

model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))       # danse를 줄여보기
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



# 모델 요약
model.summary()


# 모델 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[early_stopping])


# 모델 정확도 검증
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# 모델을 저장하려면
model.save('CNN_test/64_64_batch128_epoch8_he.h5')

# 검증데이터셋의 오차
y_vloss = hist.history['val_loss']
y_vacc = hist.history['val_accuracy']
# 학습데이터셋의 오차
y_loss = hist.history['loss']
y_acc = hist.history['accuracy']


x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='valset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


plt.plot(x_len, y_vacc, marker='.', c='red', label='valset_accuracy')
plt.plot(x_len, y_acc, marker='.', c='blue', label='Trainset_accuracy')

plt.legend(loc='lower right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()