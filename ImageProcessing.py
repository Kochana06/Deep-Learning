#importing libraries

import tensorflow as tf
from tensorflow.keras import dataset,layers,models
import matplotlib.pyplot as plt

#loading CIFAR-10 dataset

(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()


#normalization

train_images = train_images/255.0
test_images = test_images/255.0

#defining class name

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#visualizing the dataset

plt.figure(figsize(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(false)
  plt.imshow(trin_images[i],cmap = plt.cm.binary)
  plt.xlabel(class_names[train_labels[i][0]])
  plt.show

#creating sequential model

model = models.Sequential([layers.con2D(32,(3,3),activation='relu',input_shape = (32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.Flatten(),
        layers.Dense(64,activation = 'relu'),
        layers.Dense(10)
])

# printing summary

model.summary()


#compiling cnn model

model.compile(optimizer='adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
       metrics = ['accuracy'])


#training cnn model

history = model.fit(train_images,train_labels,epochs = 10,validation_data = (test_images,test_labels))

#evaluating cnn model

test_loss,test_acc = model.evaluate(test_images,test_labels,verbose = 2)
print(f' test accuracy is:{test_acc}')

#visualize the accuracy and loss values

plt.figure(figsize(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.grid()


plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.ylim([0,1])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.grid()

plt.show()


















