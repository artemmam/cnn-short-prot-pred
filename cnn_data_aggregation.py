import pandas as pd
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import json, codecs
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import sys


# ## Data-preparation

# ### Файл с преобразованной выборкой, состоящей из 8747 белков, каждый длиной L и 56 признаков
# Файл, сформированный заранее, состоящий из признаков для обучения. В качестве признаков используются аторичная структура(3), PSSM(20), FASTA-кодировка(20), растворимость(1), 8 различных типов АК по радикалу(8), 4 различных типа АК по полярности(4)

print('Loading train_data and contact matrix')
def load_data(data_name):
    pkl_file = open(data_name + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


data_train = load_data('train_data')
two_matrix = load_data('two_matrix_200')

num_classes = 2
"""
threshold = np.linspace(4, 20, num_classes - 1)
#sys.exit(0)

for i in range(len(two_matrix)):
    for j in range(len(threshold)-1):
        two_matrix[i][2][(two_matrix[i][1] <= threshold[0])] = len(threshold)
        two_matrix[i][2][((threshold[j] < two_matrix[i][1]) & (two_matrix[i][1] <= threshold[j+1]))] = j + 1
        two_matrix[i][2][(two_matrix[i][1] >= threshold[-1])] = 0
"""
print('Loaded succesfully')
# ### Файл с матрицами попарных расстояний для каждого белка из списка хороших белков
# Для всех белков из списка "хороших" белков были посчитаны опрные матрицы размерность 5 x L, которые и будут предсказываться
"""
print('Loading json')
with open("./data.json", "r") as read_file:
    data_supporting_matrix = json.load(read_file)

data_coords = pd.DataFrame()
for i in range(len(data_supporting_matrix['data'])):
    coords = pd.DataFrame({'pdb': [data_supporting_matrix['data'][i][0]], 'matrix_coord':[data_supporting_matrix['data'][i][1]]})
    data_coords =  pd.concat([data_coords, coords], ignore_index=True)
"""


good_prot = pd.read_csv('good_prot.csv')
print('Loaded succesfully')

# ### Подчищение выборки
# Была незначительная ошибка при формировании выборки, связанная с тем, что некоторые белки повторялись, в связи с чем эти повторения необходимо удалить

# In[37]:
"""
print('Some drop')
s=0
list_to_drop = []
data_target = pd.DataFrame()
for pdb in good_prot.pdb_name:
    if pdb in data_coords.pdb.tolist():
        data_target = pd.concat([data_target, data_coords.loc[data_coords['pdb'] == pdb]], ignore_index=True)
    else:
        list_to_drop.append(pdb)

target_supporting_matrix = data_target.matrix_coord.values
"""
data_features = pd.read_csv('pdb_and_features.csv')
index_to_drop = []
"""
for pdb in list_to_drop:
    index_to_drop.append(data_features.index[data_features.pdb_name == pdb].tolist())
    data_features = data_features[data_features.pdb_name != pdb]
"""


# In[44]:


for i in range(len(index_to_drop)):
    del data_train[index_to_drop[i][0]]


# In[45]:


data_features = data_features.reset_index(drop=True)


# Проверяем, всё ли теперь верно и совпадают ли размерности списков матриц для обучения и список опорных матриц  


print('Drop succesfully')
#print(len(target_supporting_matrix))
print(len(data_train))


def check_prot_size(i):
    print(data_features.iloc[i])
    print('---')
    #print(data_target.iloc[i])
    print('---')
    #print('Длина белка из матрицы расстояний',np.shape(data_target.iloc[i].matrix_coord)[0])
    print('Длина белка из выборки для обучения',np.shape(data_train[i])[0])

import sys

protein_len = []  # List of all proteins, which we have in our dataset
pdb_200 = []  # Proteins of size lower than 200


for i in range(len(data_features)):
    protein_len.append(len(data_features.FASTA[i]))
    if len(data_features.FASTA[i])<=200:
        pdb_200.append(data_features.pdb_name[i])


protein_len = np.array(protein_len)
low_border = 15  # Set lower bound for our filter
high_border = 30  # Set higher bound for our filter
protein_len_bound= protein_len[(protein_len<=high_border) & (protein_len>=low_border)]  # List of bounded protein lengths

s = 0
for i in range(len(two_matrix)):
    if (np.shape(two_matrix[i][2])[0]<=high_border) and (np.shape(two_matrix[i][2])[0]>=low_border):
        s +=1

print('Amount of proteins, which we have after bounding: ', s)


target = []

mm = high_border
for i in range(len(two_matrix)):
    if (high_border >= np.shape(two_matrix[i][2])[0]) and (np.shape(two_matrix[i][2])[0] >= low_border):
        #f = np.median(two_matrix[i][2], axis = 0) + np.zeros((mm-np.shape(two_matrix[i][2])[0], 1))
        f = np.zeros((mm-np.shape(two_matrix[i][2])[0], np.shape(two_matrix[i][2])[0]))
        f1 = np.zeros((mm, mm-np.shape(two_matrix[i][2])[0]))
        bot = np.concatenate((two_matrix[i][2], f), axis=0)
        target.append([two_matrix[i][0], np.concatenate((bot, f1), axis=1)])

train = []
mm = high_border
for i in range(len(data_train)):
    if (high_border >= np.shape(data_train[i])[0]) and (len(data_features.FASTA[i]) <= high_border) and \
            (len(data_features.FASTA[i]) >= low_border):
        #f = np.median(data_train[0], axis = 0) + np.zeros((mm-np.shape(data_train[i])[0], 1))
        f = np.zeros((mm-np.shape(data_train[i])[0], 56))
        train.append([data_features.pdb_name[i], np.concatenate((data_train[i], f), axis=0), len(data_features.FASTA[i])])
    elif (len(data_features.FASTA[i]) <= high_border) and (len(data_features.FASTA[i]) >= low_border):
        train.append([data_features.pdb_name[i], data_train[i][:mm], len(data_features.FASTA[i])])


for pdb in train:
    if pdb[0] not in np.array(target)[:, 0]:
        train.remove(pdb)

list_to_drop = []
while len(target)!=len(train):
    for pdb in target:
        if pdb[0] not in np.array(train)[:, 0]:
            target.remove(pdb)

#train = np.array(train)
#target = np.array(target)

print(np.shape(train))
print(np.shape(target))


#sys.exit(0)
train, train_val_test, train_label, target_val_test = train_test_split(train, target, test_size=0.3, random_state=13)
test, valid, test_label, valid_label = train_test_split(train_val_test, target_val_test, test_size=0.5, random_state=13)

def redo(train_X):
    out = []
    for i in range(len(train_X)):
        out.append(train_X[i][1])
    return(np.array(out))


def train_transform(x):
    x = redo(np.array(x))
    x = np.expand_dims(x, axis=4)
    return x


def reshape_of_labels(x):
    lst1 = [[ [0 for col in range(x.shape[2])] for col in range(x.shape[0])] for row in range(x.shape[1])]
    np.shape(lst1)
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                lst1[j][i][k] = x[i, j, k]
    return lst1


def target_transform(x):
    #print(x.shape)
    x = redo(np.array(x))
    x_w = x.shape[0]
    x_h = x.shape[1]
    x_l = x.shape[2]
    #print(x_w, x_h, x_l)
    x = x.reshape(x_w, x_h*x_l)
    enc = OneHotEncoder(sparse=False)
    x = enc.fit_transform(x.reshape(-1, 1))
    x = x.reshape(x_w, x_h*x_l, num_classes)
    #print(x.shape)
    x = reshape_of_labels(x)
    return x


train = train_transform(np.array(train))
test = train_transform(np.array(test))
valid = train_transform(np.array(valid))


train_label = target_transform(np.array(train_label))
test_label = target_transform(np.array(test_label))
valid_label = target_transform(np.array(valid_label))


def save_file(data, data_name):
    output = open(data_name + '.pkl', 'wb')
    pickle.dump(data, output)
    output.close()

save_file(train, 'train')
save_file(train_label, 'train_label')
save_file(test, 'test')
save_file(test_label, 'test_label')
save_file(valid, 'valid')
save_file(valid_label, 'valid_label')