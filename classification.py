
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
            batch_size=32
        )

        val_generator = train_datagen.flow_from_directory(
            self.test_path,
            target_size=(224, 224),
            batch_size=32
        )

        return train_generator, val_generator

class Load_model:
    def __init__(self, train_path):
        self.num_class = len(os.listdir(train_path)) # 클래스 수
        

    def load_weights(self, model, path_to_weights):
        """모델에 가중치를 로드하는 함수"""
        model.load_weights(path_to_weights)
        return model
    
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

        model.add(Dropout(rate=0.2))         
        model.add(Flatten())

        model.add(Dense(512,activation='relu'))
        model.add(Dense(self.num_class, activation="softmax"))
        model.summary()
        return model


class Fine_tunning:
    def __init__(self, train_path, val_path,model_name, epoch, weights_path=None):
        self.data = Import_data(train_path, val_path)
        self.train_data, self.val_data = self.data.train()
        self.load_model = Load_model(train_path)
        self.epoch = epoch
        self.model_name = model_name
        self.train_path = train_path
        self.weights_path = weights_path  # 추가: 가중치 경로 저장
    
        
        
    def training(self):
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-3]

        # 옵티마이저 정의
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.000213388)

        # 모델 생성
        model = self.load_model.build_network()
        
        if self.weights_path:
            model = self.load_model.load_weights(model, self.weights_path)

        # 학습모델 저장할 경로 생성
        save_folder = 'C:/Users/oceanlightai/Desktop/datasets/pet_skin/train/crop/' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/'
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
            callbacks=[check_point],
            verbose=1)
        return history

    def save_accuracy(self, history):
        # 학습모델 저장 경로
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-3]
        save_folder = 'C:/Users/oceanlightai/Desktop/datasets/pet_skin/train/crop/' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/'

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


train_path = './train/size_256_img' 
val_path = './valid/size_256_img'
model_name = 'size_256'
weights_path = "./ROI_model.h5"  # 저장된 모델의 경로를 지정합니다.

epoch = 100

if __name__ == '__main__':
    fine_tunning = Fine_tunning(train_path=train_path,
                                val_path=val_path,
                                model_name=model_name,
                                epoch=epoch, 
                                weights_path=weights_path)  # 가중치 경로 추가
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history) 

""" 학습 결과를 확인 해보자 """
import pandas as pd
from sklearn import metrics

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


name = 'test'
modelPath = "./ROI_model.h5" 

# weight = 'model-078-0.925417-0.916944.h5'        # 학습된 모델의 파일이름
test_Path = './test/size_256_img'


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

"""Confusion Matrix"""

from tensorflow.keras.models import load_model

# 모델 불러오기
model = load_model("./ROI_model.h5")


from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_width, img_height = 224, 224
batch_size = 32 
val_datagen = ImageDataGenerator(rescale=1./255)
val_data = val_datagen.flow_from_directory(
    './valid/size_256_img',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 예측
y_pred = model.predict(val_data, steps=val_data.samples // val_data.batch_size + 1)
y_pred_classes = np.argmax(y_pred, axis=1)

# 실제값
y_true = val_data.classes
class_labels = list(val_data.class_indices.keys())

# confusion matrix 생성
cm = confusion_matrix(y_true, y_pred_classes)

# 시각화
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()