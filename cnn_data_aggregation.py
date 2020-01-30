import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


print('Loading train_data and contact matrix')


def load_data(data_name):
    pkl_file = open(data_name + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

#  Load data and target
data_train = load_data('train_data')
two_matrix = load_data('two_matrix_200')
num_classes = 2
print('Loaded successfully')

data_features = pd.read_csv('pdb_and_features.csv')
data_features = data_features.reset_index(drop=True)
print(len(data_train))

protein_len = []  # List of all proteins, which we have in our dataset
pdb_200 = []  # Proteins of size lower than 200


low_border = 15  # Set lower bound for our filter
high_border = 30  # Set higher bound for our filter

s = 0
for i in range(len(two_matrix)):
    if (np.shape(two_matrix[i][2])[0]<=high_border) and (np.shape(two_matrix[i][2])[0]>=low_border):
        s +=1

print('Amount of proteins, which we have after bounding: ', s)


target = []


#  Apply zero-padding to make same size for all train and target samples
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

#  Check if there is any mistakes and repeated proteins
for pdb in train:
    if pdb[0] not in np.array(target)[:, 0]:
        train.remove(pdb)

list_to_drop = []
while len(target)!=len(train):
    for pdb in target:
        if pdb[0] not in np.array(train)[:, 0]:
            target.remove(pdb)
#  Check sizes of training and target sampels
print(np.shape(train))
print(np.shape(target))

train, train_val_test, train_label, target_val_test = train_test_split(train, target, test_size=0.3, random_state=13)
test, valid, test_label, valid_label = train_test_split(train_val_test, target_val_test, test_size=0.5, random_state=13)


# Some transformations to make train and target data


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
    x = redo(np.array(x))
    x_w = x.shape[0]
    x_h = x.shape[1]
    x_l = x.shape[2]
    x = x.reshape(x_w, x_h*x_l)
    enc = OneHotEncoder(sparse=False)
    x = enc.fit_transform(x.reshape(-1, 1))
    x = x.reshape(x_w, x_h*x_l, num_classes)
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