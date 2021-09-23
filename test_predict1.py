import tensorflow as tf
import numpy as np
from PIL import Image

#モデルのロード
model = tf.keras.models.load_model("MNIST_model")
image_shape = (28, 28, 1)

#サンプルデータのロード
img = np.array(Image.open('test/MNIST_sample1.png').resize(image_shape[:2])) / 255
print(img.shape)
#先頭に次元を追加
img = img[np.newaxis]
print(img.shape)
#末尾に次元を追加
img = img[:, :, :, np.newaxis]
print(img.shape)

#予測
predictions_single = model.predict(img)
print(predictions_single)
print(predictions_single.shape)
print(predictions_single[0].argmax())