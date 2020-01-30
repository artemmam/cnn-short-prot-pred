import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_name):
    pkl_file = open(data_name + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data
good = 0
bad = 0
test_label = load_data('test_label')
predicted = load_data('./predictions/predicted2')
print(np.shape(test_label))
print(np.shape(predicted))
test_label = np.array(test_label)
test = []
pred = []
for i in range(np.shape(test_label)[1]):
    for j in range(np.shape(test_label)[0]):
        test.append(np.argmax(test_label[j][i]))
        pred.append(np.argmax(predicted[j][i]))
    #if np.argmax(test_label[i] [j]) == np.argmax(predicted[i] [j]):
    #    good += 1
    #else:
    #    bad += 1
#print(good/(good + bad))
#print(np.array(test).reshape(30,30))
#print(np.array(pred).reshape(30,30))
test = np.array(test).reshape(58, 30, 30)
pred = np.array(pred).reshape(58, 30, 30)
#print(test[0])
for k in range(58):
    for i in range(30):
        for j in range(30):
            if test[k,i,j] == pred[k,i,j] and test[k,i,j] == 1:
                good += 1
            elif test[k,i,j] == 1 and test[k,i,j] != pred[k,i,j]:
                bad +=1
print(good/(bad+good))

prot_number = 35
predicted = np.array(predicted)
visual_pred = pred[prot_number]
visual_test = test[prot_number]
#visual_test = visual_test.reshape(900)
#visual_pred = visual_pred.reshape(900)
#print(visual_pred)
visual_pred[(visual_pred != 0) & (visual_pred <= 16) | (visual_pred == 63)] = 1
visual_test[(visual_test != 0) & (visual_test <= 16) | (visual_test == 63)] = 1
#print(visual_pred)
visual_pred[(visual_pred>16) & (visual_pred!=63)] = 0
visual_test[(visual_test>16) & (visual_test!=63)] = 0

#visual_test.reshape(30, 30)
#visual_pred.reshape(30, 30)
#visual_pred_alt = predicted_final_alt[prot_number]
fig, (ax1, ax2)=plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
#fig.add_subplot(1, 1, 1)
ax1.set_title('Test')
ax1.imshow(visual_test, cmap='rainbow')
for i in range(30):
        for j in range(30):
            text = ax1.text(j, i, visual_test[i, j],
                           ha="center", va="center", color="black")
ax2.set_title('Predicted')
ax2.imshow(visual_pred, cmap='rainbow')
for i in range(30):
        for j in range(30):
            text = ax2.text(j, i, visual_pred[i, j],
                           ha="center", va="center", color="black")
#for i in range(30):
#    for j in range(30):
#        text = ax1.text(j, i, test[i, j], ha="center", va="center", color="black")
#plt.imshow(visual_pred_alt, alpha=.8)
plt.show()
import sys
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
sys.exit(0)









def reshape_of_reshape_labels(x):
    lst1 = [[ [0 for col in range(np.array(x).shape[2])] for col in range(np.array(x).shape[0])] for row in range(np.array(x).shape[1])]
    print(np.shape(lst1))
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




"""
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