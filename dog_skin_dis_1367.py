#!/usr/bin/env python
# coding: utf-8

# In[1]:


#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt 

"""전처리를 위한 라이브러리"""
import os
import pandas as pd
import numpy as np
import cv2
import random
"""Keras 라이브러리"""
import tensorflow.keras as keras 
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import * 
from tensorflow.keras import Sequential 
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.models import load_model
"""confusion matrix 라이브러리"""
from sklearn.metrics import confusion_matrix
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"]="0"


class Import_data:
    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.test_path = val_path

    def train(self):
        # generator 생성
        train_datagen=ImageDataGenerator(rescale=1. / 255,
                                           featurewise_std_normalization=True,
                                           zoom_range=0.2,
                                           channel_shift_range=0.1,
                                           rotation_range=20,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           horizontal_flip=True                             
                                        )
        
        val_datagen=ImageDataGenerator(
                                        rescale=1.0/255,
                                        rotation_range = 20
                                        
                                       )
        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(224, 224),
            batch_size=16
        )

        val_generator = train_datagen.flow_from_directory(
            self.test_path,
            target_size=(224, 224),
            batch_size=16
        )

        return train_generator, val_generator



class Load_model:
    def __init__(self, train_path):
        self.num_class = len(os.listdir(train_path))  
    
    def build_network(self):
        model=Sequential()

        model.add(Conv2D(32,(2,2),input_shape=(224,224,3),activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,(2,2),activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(128,(2,2),activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(256,(2,2),activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(512,(2,2),activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

#         model.add(Conv2D(1024,(2,2),input_shape=(224,224,3),activation="relu"))
#         model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(rate=0.2))         
        model.add(Flatten())

        model.add(Dense(512,activation='relu'))
        model.add(Dense(self.num_class, activation="softmax"))
        model.summary()
        return model

    
    def load_trained_weights(self, model, weight_path):
        model.load_weights(weight_path)
        return model



class Fine_tunning:
    def __init__(self, train_path, val_path, model_name, epoch):
        self.data = Import_data(train_path, val_path)
        self.train_data, self.val_data = self.data.train()
        self.load_model = Load_model(train_path)
        self.epoch = epoch
        self.model_name = model_name
        self.train_path = train_path
    def training(self, pretrained_weights=None):
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-3]

        # 옵티마이저 정의
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.000213388)
        
        # 모델 생성
        model = self.load_model.build_network()

        # 저장된 가중치가 제공되면 로드
        if pretrained_weights:
            model = self.load_model.load_trained_weights(model, pretrained_weights)
        
        # 학습모델 저장할 경로 생성
        save_folder = '저장위치' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # 훈련 중 주기적으로 모델 저장
        check_point = ModelCheckpoint(save_folder + 'model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                                      monitor='val_acc', save_best_only=True, mode='auto')                            
        # 모델 컴파일
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['acc'])

        # 모델 학습
        history = model.fit(
            self.train_data,
            steps_per_epoch=self.train_data.samples / self.train_data.batch_size,
            epochs=self.epoch,
            validation_data=self.val_data,
            validation_steps=self.val_data.samples / self.val_data.batch_size,
            callbacks=[check_point]
            ,
            #es],
            verbose=1)
        return history
    
    
    def save_accuracy(self, history):
        # 학습모델 저장 경로
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-3]
        save_folder = '저장위치' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/'

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        epoch_list = list(epochs)

        # csv 저장
        df = pd.DataFrame({'epoch': epoch_list, 'train_accuracy': acc, 'validation_accuracy': val_acc},
                          columns=['epoch', 'train_accuracy', 'validation_accuracy'])
        df_save_path = save_folder + 'accuracy.csv'
        df.to_csv(df_save_path, index=False, encoding='euc-kr')

        # Accuracy 그래프 이미지 저장
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        save_path = save_folder + 'accuracy.png'
        plt.savefig(save_path)
        plt.cla()

        # Loss 그래프 이미지 저장
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        save_path = save_folder + 'loss.png'
        plt.savefig(save_path)
        plt.cla()

        # 마지막 모델을 제외하고 삭제
        name_list = os.listdir(save_folder)
        h5_list = []
        for name in name_list:
            if '.h5' in name:
                h5_list.append(name)
        h5_list.sort()
        h5_list = [save_folder + name for name in h5_list]
        for path in h5_list[:len(h5_list) - 1]:
            os.remove(path)
        K.clear_session()
        
        
        
    def visualize_results(self):

        model = self.load_model.build_network()  # 모델 구조 불러오기

        # 마지막으로 저장된 모델 불러오기
        save_folder = '모델 저장위치' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/'
        model_files = [f for f in os.listdir(save_folder) if f.endswith('.h5')]
        model_files.sort()
        latest_model_file = os.path.join(save_folder, model_files[-1])
        model.load_weights(latest_model_file)  # 가장 최근 모델의 weights를 불러옵니다.

        # 예측 수행
        y_pred = model.predict(self.val_data)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = self.val_data.classes

        # Confusion Matrix 생성
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

        # 클래스 레이블 추출
        class_labels = list(self.val_data.class_indices.keys())

        # Confusion Matrix 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("예측된 라벨")
        plt.ylabel("실제 라벨")
        plt.title("Confusion Matrix")
        plt.show()

        # Classification Report 출력
        print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))


# In[5]:


# from ops import *
train_path = '/train' 
val_path = '/valid'
model_name = 'model'
epoch = 50


if __name__ == '__main__':
    fine_tunning = Fine_tunning(train_path=train_path,
                                val_path=val_path,
                                model_name=model_name,
                                epoch=epoch)
    # 이전에 학습했던 모델의 가중치 파일 경로
    pretrained_weights_path = '/model.h5'  
    history = fine_tunning.training(pretrained_weights=pretrained_weights_path)
    fine_tunning.save_accuracy(history)
    fine_tunning.visualize_results()


# In[7]:


import pandas as pd
from sklearn import metrics

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


name = 'lesion'
modelPath = "/model.h5"

# weight = 'model-078-0.925417-0.916944.h5'        # 학습된 모델의 파일이름
test_Path = '/test'


# model = load_model(modelPath + weight)
model = load_model(modelPath)
datagen_test = ImageDataGenerator(rescale=1./255)

generator_test = datagen_test.flow_from_directory(directory=test_Path,
                                                  target_size=(224, 224),
                                                  batch_size=16,
                                                  shuffle=False)

# model로 test set 추론
generator_test.reset()
cls_test = generator_test.classes
cls_pred = model.predict(generator_test, verbose=1, workers=0)
cls_pred_argmax = cls_pred.argmax(axis=1)

# 결과 산출 및 저장
report = metrics.classification_report(y_true=cls_test, y_pred=cls_pred_argmax, output_dict=True)
report = pd.DataFrame(report).transpose()
#report.to_csv(f'e:/output/report_test_{name}.csv', index=True, encoding='cp949')
report.to_csv(f'report_test_{name}.csv', index=True, encoding='cp949')
print(report)


# In[8]:


import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('/model.h5')

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.
    
    prediction = model.predict(img_array)
    return prediction[0]

def diagnose_image(image_path):
    classes = ["A1", "A3", "A6", "A7"]
    prediction = predict_image(image_path)
    
    # 예측 결과에서 가장 높은 확률을 가진 클래스의 인덱스를 가져옴
    predicted_index = np.argmax(prediction)
    
    # A7은 정상
    if classes[predicted_index] == "A7":
        print("정상")
    else:
        print(f"질병: {classes[predicted_index]}")


# In[10]:


diagnose_image("img.jpg")




