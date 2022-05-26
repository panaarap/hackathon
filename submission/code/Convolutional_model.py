import ifxdaq
import processing
import numpy as np
from PIL import Image
import statistics
import tensorflow as tf
from tensorflow.keras.layers import LSTM
import os
import zipfile
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Conv3D,MaxPooling3D,Conv1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler

checkpoint_path='C:\\Users\\jimmy\\PycharmProjects\\hackathon\\examples\\weights\\weights-best.hdf5'


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

number_of_frames = 6000


folder_dir = "C:\\Users\\jimmy\Downloads\\challenge\\extended_data_and_more\\examples\\processed"

X = []
count=0

for x in os.listdir(folder_dir):
        path = folder_dir + '/' + x
        X.append(np.load(path))



sample = []


for s in range(4,number_of_frames):
            sample.append(X[s][:][:][:] - X[s-4][:][:][:])



sample=np.array(sample)

X = sample

# X = np.real(X)

y1 = np.full(1499,1)
y2 = np.full(1499,2)
y3 = np.full(1499,3)
y0 = np.full(1499,0)

y0 =y0.reshape(-1,1)
y1 =y1.reshape(-1,1)
y2 =y2.reshape(-1,1)
y3 =y3.reshape(-1,1)
print(y0.shape)


mean_antenna = []



# X = np.real(X)
print(X.shape)
print("mean")


Y = np.concatenate((y0,y1,y2,y3))

print(X.shape)

from tensorflow.keras.utils import to_categorical
# print(Y)

# encoder = OneHotEncoder(sparse=False)
# Y = encoder.fit_transform(Y)
Y=to_categorical(Y)
print(Y)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,
                                                                                            Y,
                                                                                            test_size=0.2,
                                                                                            shuffle=True,
                                                                                            random_state=39)


print(X_train.shape)

model = Sequential()

activation='relu'
model.add(MaxPooling2D(input_shape=[3,64,64]))
model.add((Conv1D(64, 3, activation = activation)))

model.add((BatchNormalization)())
model.add((Conv1D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform')))
model.add(BatchNormalization())
model.add((Conv1D(16, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform')))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(64, activation = activation, kernel_initializer = 'he_uniform'))
model.add(Dense(4, activation = 'softmax'))

model.add(Dense(4, activation='softmax'))

Adam = tf.keras.optimizers.Adam(learning_rate=0.0003) #learning rate

model.compile(optimizer = Adam ,loss = 'categorical_crossentropy', metrics = ['accuracy'] )
print(model.summary())
# model.load_weights(checkpoint_path)

# #=================================train model==================================

#Csv logger
log_csv = CSVLogger('my_logs.csv', separator=',', append=False)
#ModelCheckpoint callback saves best model
checkpoint = ModelCheckpoint("C:\\Users\\jimmy\\PycharmProjects\\hackathon\\examples\\weights-best-60acc.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#Early stopping
early_stop = EarlyStopping(monitor="val_loss", patience =3 , verbose=1)
#callbacks list
callbacks_list=[early_stop , log_csv, checkpoint]


history = model.fit(X_train, y_train, epochs=8, validation_data=(X_test, y_test),callbacks=callbacks_list)

model.save('C:\\Users\\jimmy\\PycharmProjects\\hackathon\\examples\\weights\\newreal_model')
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'm', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training & validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'm', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training & validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
