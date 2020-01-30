import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle

#  Read fasta-files
with open('./train/train.fasta', 'r') as myfile:
    data = myfile.read().replace('\n', '')
#  Create dictionaries for classifying all aminoacids according to classe from wiki (polarity and radicals)
rad_dic = {'G': '0', 'L': '0', 'Y': '1', 'S': '2', 'E': '3', 'Q': '4', 'D': '3', 'N': '4', 'F': '1',
       'A': '0', 'K': '5', 'R': '5', 'H': '6', 'C': '7', 'V': '0', 'P': '6', 'W': '6', 'I': '0', 'M': '7', 'T': '2'}
pol_dic = {'G': '0', 'L': '0', 'Y': '1', 'S': '1', 'E': '2', 'Q': '1', 'D': '2', 'N': '1', 'F': '0',
        'A': '0', 'K': '1', 'R': '1', 'H': '3', 'C': '0', 'V': '0', 'P': '0', 'W': '0', 'I': '0', 'M': '0', 'T': '1'}

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

bad_prot = []
pdb_name = ''
seq = ''
j = 0
fasta_df = pd.DataFrame()
mm = 0

#  Record protein name, fasta sequences, radical and polarity classes into dataframe
for i in range(len(data)-1):
    seq = ''
    if data[i] == '$' and data[i+5] == '%':
        pdb_name = data[i+1:i+5]
    if data[i] == '%':
        j = i+1
        while data[j] != '$' and j != len(data)-1:
            if data[j] not in rad_dic:
                if pdb_name not in bad_prot:
                    bad_prot.append(pdb_name)
            seq += data[j]
            j = j+1
            if j == len(data)-1:
                seq += data[j]
        if len(seq) > mm:
            mm = len(seq)
        replace_all(seq, rad_dic)
        temp_df = pd.DataFrame({'pdb_name': [pdb_name], 'FASTA':[seq], 'Radical':[replace_all(seq, rad_dic)]
                           , 'Polarity':[replace_all(seq, pol_dic)]})
        fasta_df = pd.concat([fasta_df, temp_df], ignore_index=True)
print("Record FASTA df: OK")

#  Read PSSM matrix and also record it into dataframe
file = open('./train/train.pssm', 'r')
pssm = file.readlines()
rep = {'G ':'', 'L ':'', 'Y ':'', 'S ':'', 'E ':'', 'Q ':'', 'D ':'', 'N ':'', 'F ':'', 
       'A ':'', 'K ':'', 'R ':'', 'H ':'', 'C ':'', 'V ':'', 'P ':'', 'W ':'', 'I ':'', 'M ':'', 'X ':'', 'T ':'', '\n':''}


def string_to_float(a):
    s1 = replace_all(a, rep)
    s1 = s1.split(' ')
    s1 = [float(i) for i in s1]
    return s1


pssm_df = pd.DataFrame()
gd = []
j = 0
for i in range(len(pssm)-1, -1, -1):
    if pssm[i][0]=='>':
        gd = list(reversed(gd))
        temp_df = pd.DataFrame({'PSSM': [gd]})
        pssm_df = pd.concat([pssm_df, temp_df], ignore_index=True)
        gd = []
    if pssm[i][0]!='>':
        gd.append(string_to_float(pssm[i]))
print("Record PSSM df: OK")

pssm_df = pssm_df.iloc[::-1]
pssm_df = pssm_df.reset_index(drop=True)

# Read solvent accessibility and secondary structure classes
file = open('./train/train.acc', 'r')
acc = file.readlines()

file = open('./train/train.ss', 'r')
ss = file.readlines()

ss_acc_df = pd.DataFrame()
for i in range(len(ss)):
    if i%2 != 0:
        temp = pd.DataFrame({'SS': [ss[i].replace('\n', '')], 'ACC':[acc[i].replace('\n', '')]})
        ss_acc_df = pd.concat([ss_acc_df, temp], ignore_index=True)
print("Record SS and ACC df: OK")
# Concatenate all dataframes into one
result = pd.concat([fasta_df, ss_acc_df, pssm_df], axis=1, sort=False)
print("Concatenate dfs: OK")
#  Delete proteins with untypical aminoacids in sequence
bad = []
for i in range(len(result)):
    if result.pdb_name.iloc[i] in bad_prot:
        bad.append(i)
result = result.drop(bad)
result = result.reset_index(drop=True)
print("Drop bad proteins: OK")

result.pdb_name.to_csv('good_prot.csv', index=False)
result.to_csv('pdb_and_features.csv', header='pdb_name')


#  Transform categorical data into binarized labels in a one-vs-all fashion
asd = []
for i in range(len(result)):
    fas = list(result.FASTA[i])
    ss1 = list(result.SS[i])
    acc1 = list(result.ACC[i])
    pol = list(result.Polarity[i])
    rad = list(result.Radical[i])
    lb = preprocessing.LabelBinarizer()
    lb1 = preprocessing.LabelBinarizer()
    lb2 = preprocessing.LabelBinarizer()
    lb3 = preprocessing.LabelBinarizer()
    lb4 = preprocessing.LabelBinarizer()
    lb.fit(['G', 'L', 'Y', 'S', 'E', 'Q', 'D', 'N', 'F', 'A', 'K', 'R', 'H', 'C', 'V', 'P', 'W', 'I', 'M', 'T'])
    a = lb.transform(fas)
    lb1.fit(['C', 'H', 'E'])
    b = lb1.transform(ss1)
    lb3.fit(['e', '-'])
    c = lb3.transform(acc1)
    lb2.fit(['0', '1', '2', '3', '4', '5', '6', '7'])
    d = lb2.transform(rad)
    lb4.fit(['0', '1', '2', '3'])
    e = lb4.transform(rad)
    pdb1 = np.concatenate((a, b, c, d, e, result.PSSM[i]), axis=1)
    asd.append(pdb1)
#  Save final data into pickle file
print("Binarized labels: OK")
output = open('train_data.pkl', 'wb')
pickle.dump(asd, output)
output.close()
print("Record data: OK")

