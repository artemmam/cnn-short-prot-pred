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

reg = l2(l2_lambda)
init="he_normal"

mm1 = 30
mm = 30
#1 слой
#model.add(Embedding(100, 8, input_length=mm))
print('Creating CNN for prediction')


input_layer = Input(shape=(mm, 56, 1))
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


print('Compiling model...')
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.001))#, metrics=['msle', 'mae', coeff_determination])


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