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

prot_number = 44
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
    visual_pred = pred[prot_number]
    visual_test = test[prot_number]
    #visual_pred_alt = predicted_final_alt[prot_number]
    fig=plt.figure(figsize=(12, 12))
    fig.add_subplot(1, 1, 1)
    plt.imshow(visual_test, cmap='Paired')
    plt.imshow(visual_pred, alpha=.5, cmap='gray')
    #plt.imshow(visual_pred_alt, alpha=.8)
    plt.show()

visualize(prot_number)
