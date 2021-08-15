
# model.py

from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Input, Dropout
from keras import optimizers
import matplotlib.pyplot as plt
from split import *

# define input_tensor
input_tensor = Input(shape=(64,64,3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(64, activation='sigmoid'))
top_model.add(Dropout(0.5))
top_model.add(Dense(32, activation='sigmoid'))
top_model.add(Dropout(0.5))
top_model.add(Dense(6, activation='softmax'))

model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# vgg_model apply to 15layers
for layer in model.layers[:15]:
    layer.trainable = False

# compile
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
history = model.fit(train_X_scratch, train_y_scratch, epochs=100, batch_size=32, validation_data=(test_X, test_y))
score = model.evaluate(test_X, test_y, verbose=0)
print(score)

# save model
model.save('my_model.h5')

# plot acc, val_acc
plt.plot(history.history['acc'], label='acc', ls='-')
plt.plot(history.history['val_acc'], label='val_acc', ls='-')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()


