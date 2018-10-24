# the_Second_Test
keras卷积神经网络识别cifar-10图像

#######数据预处理#################

#将features标准化
x_img_train_normalize=x_img_train.astype('float32')/255.0
x_img_test_normalize=x_img_test.astype('float32')/255.0

#label(照片图像真实的值)以一位有效编码进行转换
from keras.utils import np_utils
y_label_train_onehot=np_utils.to_categorical(y_label_train)
y_label_test_onehot=np_utils.to_categorical(y_label_test)

#########建立模型###################

##建立3次的卷积运算神经网络

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

#建立卷积层2与池化层2
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

#建立卷积层3与池化层3
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))

#建立神经网络
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(2500,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1500,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))

#######################进行训练##############################

##进行训练
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x_img_train_normalize,y_label_train_onehot,validation_split=0.2,epochs=10,batch_size=128,verbose=1)

#####################评估模型准确率###############################

scores=model.evaluate(x_img_test_normalize,y_label_test_onehot,verbose=0)
scores[1]

#####################进行预测#######################################

prediction=model.predict_classes(x_img_test_normalize)
prediction[:10]
 
#####################查看预测概率#################################

predicted_probability=model.predict(x_img_test_normalize)

#####################显示混淆矩阵################################

import pandas as pd
pd.crosstab(y_label_test.reshape(-1),prediction,rownames=['label'],colnames=['predict'])

#####################模型的保存###################################

model.save_weights("SaveModel/cifarCnnModel.h5")

