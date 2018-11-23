import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist  # 28x28 images og hand written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data() #load the MNIST dataset using the Keras helper function
x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)    # scales data between 0 and 1

model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',# Good default optimizer to start with
              loss='sparse_categorical_crossentropy', # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy']) # what to track
model.fit(x_train, y_train, epochs=3) # train the model

val_loss, val_acc = model.evaluate(x_test, y_test) # evaluate the out of sample data with model
print(val_loss, val_acc) # model's loss (error) & accuracy

plt.subplot(221)
plt.imshow(x_test[0],cmap=plt.cm.binary) # prediction example
plt.subplot(222)
plt.imshow(x_test[1],cmap=plt.cm.binary) # prediction example
plt.subplot(223)
plt.imshow(x_test[3],cmap=plt.cm.binary) # prediction example
plt.subplot(224)
plt.imshow(x_test[4],cmap=plt.cm.binary) # prediction example
plt.show()
