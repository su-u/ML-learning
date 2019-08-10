import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
mnist = tf.keras.datasets.mnist

#%%
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [28, 28])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)
#%%


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

table = []
for x in y_train:
    if x in table:
        pass
    else:
        table.append(x)

print(table)
#%%
print(x_train.shape)
print(x_train[0])

#%%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)
print(model.evaluate(x_test, y_test))

#%%
plt.figure()
plt.imshow(x_train[0])
plt.show()
predictions = model.predict(x_test)

plt.figure()
plt.imshow(x_test[0])
plt.show()
print(predictions[0])

#%%
num_rows = 5
num_cols = 5
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, y_test, x_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, y_test)
plt.show()

#%%
image_r = tf.io.read_file('2-2.jpg')
image = tf.image.decode_image(image_r, 1)
image_j = tf.image.decode_jpeg(image_r, 1)

image_f = load_and_preprocess_image('2-2.jpg')
image_f = tf.image.rgb_to_grayscale(image_f)
image_f = tf.squeeze(tf.cast(255.0 - (image_f * 255.0), tf.uint8))

plt.figure()
plt.imshow(image_f)
plt.show()




img = (np.expand_dims(image_f, 0))
predictions_s = model.predict(img)
print(predictions_s)
predicted_label = np.argmax(predictions_s)
print(predicted_label)

