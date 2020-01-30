import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, LSTM, Embedding, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2 # L2-regularisation
from keras.models import Model
import tensorflow as tf
import warnings


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


warnings.filterwarnings("ignore")


l2_lambda = 0.0001


def load_data(data_name):
    pkl_file = open(data_name + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

print('Loading data...')
train = load_data('train')
train_label = load_data('train_label')
test = load_data('test')
test_label = load_data('test_label')
valid = load_data('valid')
valid_label = load_data('valid_label')
print('Data are loaded')


# ## CNN для предсказание опорной матрицы
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


batch_size = 4
epochs = 25
#def coeff_determination(y_true, y_pred):
#    from keras import backend as K
#    SS_res =  K.sum(K.square( y_true-y_pred ))
#    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
#    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# ### Созание модели для обучения

reg = l2(l2_lambda)
init="he_normal"

mm1 = 30
mm = 30
#1 слой
#model.add(Embedding(100, 8, input_length=mm))
print('Creating CNN for prediction')
input_layer = Input(shape=(mm, 56, 1))
#Input(Conv2D(20, kernel_size=(mm, 5),activation='linear',padding='same', W_regularizer=l2(l2_lambda), input_shape=(mm, 44, 1)))
#Блок residual network

#tower_1 = (Conv2D(32, (1, 1), padding="same", kernel_initializer=init, kernel_regularizer=reg))(input_layer)
#tower_1 = (LeakyReLU(alpha=0.1)) (tower_1)
#tower_1 = (Conv2D(32, (1, 1), padding="same",kernel_initializer=init, kernel_regularizer=reg))(tower_1)
#tower_1 = (LeakyReLU(alpha=0.1)) (tower_1)
#tower_1 = (BatchNormalization())(tower_1)
#tower_1 = (MaxPooling2D(pool_size=(2, 2),padding='same'))(tower_1)
#tower_1 = (Dropout(0.1))(tower_1)
print('')
num_classes = np.shape(train_label)[2]
layer1 = (Conv2D(8, (mm, 5), activation='linear', W_regularizer=l2(l2_lambda), padding='same'))(input_layer)
layer1 = (LeakyReLU(alpha=0.1))(layer1)
layer1 = (Conv2D(16, (mm, 5), activation='linear', W_regularizer=l2(l2_lambda), padding='same'))(layer1)
layer1 = (LeakyReLU(alpha=0.1))(layer1)
layer1 = (MaxPooling2D(pool_size=(2, 2),padding='same'))(layer1)
layer1 = (Dropout(0.1))(layer1)

layer2 = (Conv2D(8, (mm, 5), activation='linear', W_regularizer=l2(l2_lambda), padding='same'))(layer1)
layer2 = (LeakyReLU(alpha=0.1))(layer2)
layer2 = (Conv2D(16, (mm, 5), activation='linear', W_regularizer=l2(l2_lambda), padding='same'))(layer2)
layer2 = (LeakyReLU(alpha=0.1))(layer2)
layer2 = (MaxPooling2D(pool_size=(2, 2),padding='same'))(layer2)
layer2 = (Dropout(0.1))(layer2)


layer3 = (Conv2D(8, (mm, 5), activation='linear', W_regularizer=l2(l2_lambda), padding='same'))(layer2)
layer3 = (LeakyReLU(alpha=0.1))(layer3)
layer3 = (Conv2D(16, (mm, 5), activation='linear', W_regularizer=l2(l2_lambda), padding='same'))(layer3)
layer3 = (LeakyReLU(alpha=0.1))(layer3)
layer3 = (MaxPooling2D(pool_size=(2, 2),padding='same'))(layer3)
layer3 = (Dropout(0.1))(layer3)

layer4 = (Flatten())(layer3)
layer4 = (BatchNormalization())(layer4)
layer4 = (Dense((5 * mm1), activation='linear', W_regularizer=l2(l2_lambda)))(layer4)
layer4 = (LeakyReLU(alpha=0.1))(layer4)
layer4 = (Dropout(0.1))(layer4)
finish = []

for i in range(mm1**2):
    output = Dense(np.shape(train_label)[2], activation='softmax', name='main_output'+str(i))(layer4)
    finish.append(output)


model = Model(inputs=[input_layer], outputs=finish)


# 1) Батч-нормализация (не) нужна
# 2) Чем меньше данныых тем меньше слоёв лучше использовать
# 3) При уменьшении слоёв происходит переобучение, мб уменьшить лёрнинг рейт
# 4) трёх слоёв достаточно
# 5) батч-сайз = 16

# In[18]:


#optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


# Компилируем модель. В качетсве оптимизатора используем обычный ADAM, ф-ция поетрь -- среднеквадратичная средняя ошибка, метрики -- средняя абсолютная ошибка и коэфициент детерминации R^2

# In[19]:


#def auc(y_true, y_pred):
#    auc = tf.metrics.auc(y_true, y_pred)[1]
#    K.get_session().run(tf.local_variables_initializer())
#    return auc

print('Compiling model...')
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.001))#, metrics=['msle', 'mae', coeff_determination])


# Обучаем модель
#from keras_tqdm import TQDMCallback
print('Start training...')
train_dropout = model.fit(train, train_label, batch_size=batch_size, epochs=epochs, verbose=0,
                          validation_data=(valid, valid_label))
print('Model has trained!')
print('Start predicting...')
predicted = model.predict(test)
print('Predicted successfully!')
print(np.shape(predicted))
print(np.shape(test_label))
output = open('./predictions/predicted' + str(num_classes) + '.pkl', 'wb')
pickle.dump(predicted, output)
output.close()
#model.save_weights("model_cnn_cm_pred.h5")
#print("Saved model to disk")
#import sys
#sys.exit(0)
"""
def reshape_of_reshape_labels(x):
    lst1 = [[ [0 for col in range(np.array(x).shape[2])] for col in range(np.array(x).shape[0])] for row in range(np.array(x).shape[1])]
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                lst1[j][i][k] = x[i][j][k]
    return lst1


predicted = np.array(reshape_of_reshape_labels(predicted))
predicted_cm = predicted.reshape(predicted.shape[0]*predicted.shape[1], 8)
test_label = np.array(reshape_of_reshape_labels(test_label))
test_label_cm = test_label.reshape(test_label.shape[0]*test_label.shape[1], 8)


def sli(pred):
    gd = []
    for i in range(len(pred)):
        if abs(pred[i][0] - pred[i][1])>0.05:
            gd.append(np.argmax(pred[i]))
        else:
            gd.append(-1)
    return np.array(gd)


predicted_final = []
predicted_final_alt = []
for k in range(len(predicted)):
    predicted_final.append(np.argmax(predicted[k], axis=1).reshape(30, 30))
    predicted_final_alt.append(sli(predicted[k]).reshape(30, 30))


predicted_final = np.array(predicted_final)
predicted_final_alt = np.array(predicted_final_alt)

test_new = []
for k in range(len(test_label)):
    test_new.append(np.argmax(test_label[k], axis=1).reshape(30, 30))
test_new = np.array(test_new)


print('Start recording...')
output = open('predicted_cm_12_alt.pkl', 'wb')
pickle.dump(predicted_final_alt, output)
output.close()

output = open('predicted_cm_12.pkl', 'wb')
pickle.dump(predicted_final, output)
output.close()

output = open('cm_test_12.pkl', 'wb')
pickle.dump(test_new, output)
output.close()
print('Recorded successfully')
"""
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(test_label_cm, axis = 1), np.argmax(predicted_cm, axis = 1))
print(cm)


from pycm import *
cm = ConfusionMatrix(actual_vector=np.argmax(test_label_cm, axis = 1), predict_vector=np.argmax(predicted_cm, axis = 1))
print(cm)


def visualize(prot_number):
    visual_pred = np.argmax(predicted[prot_number], axis=1).reshape(30, 30)
    visual_test = np.argmax(test_label[prot_number], axis=1).reshape(30, 30)
    visual_pred_alt = predicted_final_alt[prot_number]
    fig=plt.figure(figsize=(12, 12))
    fig.add_subplot(1, 1, 1)
    plt.imshow(visual_test, cmap='Paired')
    plt.imshow(visual_pred, alpha=.5, cmap='gray')
    plt.imshow(visual_pred_alt, alpha=.8)
    plt.show()


visualize(12)
"""
#cm[0, 0]/(cm[0, 0] + cm[1, 0])#true positive(обнаружение нулей)полнота нулей

#cm[1, 0]/cm[0, 0]#false negative rate(ошибка при предсказание нулей)


#recall_1 = cm[1, 1]/(cm[0, 1] + cm[1, 1])#true negative(обнаружение единиц)полнота единиц

#print(recall_1)

#cm[0, 1]/(cm[0, 1] + cm[1, 1])#(false positive rate)(ошибка при предсказании единиц)

#cm[0, 0]/(cm[0, 0] + cm[0, 1])#positive predicted value, precision точность нулей

#prec_1 = cm[1, 1]/(cm[1, 1] + cm[1, 0])#negative predicteve value точность единиц
#print(prec_1)

#F = 2 * recall_1 * prec_1/(recall_1 + prec_1)
#print(F)

#tot_fal = 0
#tot_one = 0
#good_pred = []
#for j in range(len(pred)):
#    fal = 0
#    ones_test = np.sum(np.argmax(test_label[j], axis=1) == 1)
#    for i in range(len(np.argmax(pred[j], axis=1))):
#        if np.argmax(test_label[j], axis=1)[i] == 1 and np.argmax(pred[j], axis=1)[i] != 1:
#            fal+=1
#       # if np.argmax(test_label[j], axis=1)[i] == 0 and np.argmax(pred[j], axis=1)[i] != 0:
#       #     fal+=1
#    if fal/ones_test<0.3:
#        good_pred.append(j)
#    tot_fal = tot_fal + fal
#    tot_one = tot_one + ones_test

#tot_fal


#good_pred



"""
prot_vis = 12
visual_pred = np.argmax(pred[prot_vis], axis=1).reshape(30, 30)
visual_test = np.argmax(test_label[prot_vis], axis=1).reshape(30, 30)
visual_pred_alt = pred_new_alt[prot_vis]
fig=plt.figure(figsize=(12, 12))
fig.add_subplot(1, 1, 1)
plt.imshow(visual_test, cmap='Paired')
plt.imshow(visual_pred, alpha=.5, cmap='gray')
plt.imshow(visual_pred_alt, alpha=.8)
plt.show()


prot_vis = 35
visual_pred = pred_new_alt[prot_vis]
visual_test = np.argmax(test_label[prot_vis], axis=1).reshape(30, 30)
fig=plt.figure(figsize=(12, 12))
fig.add_subplot(1, 1, 1)
plt.imshow(visual_test, cmap='Paired')
plt.imshow(visual_pred, alpha=.5, cmap='gray')
plt.show()
"""




