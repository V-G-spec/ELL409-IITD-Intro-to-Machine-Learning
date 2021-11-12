# %%
import pandas as pd
# !pip install keras
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

# %%
split_frac = 8

df = pd.read_csv('2019EE10143.csv',header=None)
data = np.array(df.values)
sz = len(data) ;
le = (len(data)*split_frac)//10

train_data = data[:le,:784]
train_label = data[:le,784] 
train_label = np_utils.to_categorical(train_label)

test_data = data[le:,:784]
test_label = data[le:,784]
test_label = np_utils.to_categorical(test_label)

print(train_data.shape, test_data.shape)

# %%
def get_model():
    sgd  = tf.keras.optimizers.SGD(lr=0.1)
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['mean_squared_error'])
    return model

# %%
model = get_model()
model.fit(train_data, train_label, batch_size=30, epochs=50, verbose = 2)

# %%
y_pred = model.predict(test_data)
test_acc = np.sum(np.argmax(y_p)==np.argmax(test_l) for (y_p, test_l) in zip(y_pred, test_label))
test_acc /= len(test_label)
print(test_acc)

# %%
y_pred = model.predict(train_data)
train_acc = np.sum(np.argmax(y_p)==np.argmax(train_l) for (y_p, train_l) in zip(y_pred, train_label))
train_acc /= len(train_label)
print(train_acc)

# %%
true_labels = []
predicted_labels= []
for y_p in y_pred:
    predicted_labels = np.append(predicted_labels, np.argmax(y_p))
for y_p in test_label:
    true_labels = np.append(true_labels, np.argmax(y_p))
assert(true_labels.shape == predicted_labels.shape)

# %%
lebels = [0, 1, 2, 3,4,5,6,7,8,9]
cm = confusion_matrix(true_labels, predicted_labels, labels=lebels)
print(classification_report(true_labels, predicted_labels))
print("Confusion matrix")
print(cm)

# %%
def showImage(img, pred, true=None):
    plt.figure(figsize=(1,1))
    picAr = np.array(img, dtype='float')
    roughSd = int(math.sqrt(img.size))
    assert(roughSd==28)
    pic = picAr.reshape((roughSd, roughSd)).T
    plt.imshow(pic) #cmap='grey'
    lbp = str(int(pred))
    if (true!=None):
        lb = str(int(true))
        plt.title('True label is: %s \n Predicted label is: %s'%(lb, lbp))
    else:
        plt.title('Predicted label is: %s'%(lbp))
    
    plt.show()

# %%
for i, y_p in enumerate(predicted_labels):
    if (y_p!=true_labels[i]):
        showImage(test_data[i], y_p, true_labels[i])