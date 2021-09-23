import tensorflow as tf
import numpy as np
from PIL import Image

#モデルのロード
model = tf.keras.models.load_model("My_model-opt2")
# image_shape = (28, 28, 1)

# # 画像読み込み
# my_image = Image.open(request.data)
# # RGB変換
# my_image = my_image.convert('RGB')
# # リサイズ
# my_image = my_image.resize((100, 100))
# # 画像から配列に変換
# my_data = np.asarray(my_image)

# #サンプルデータのロード
# img = np.array(Image.open('test/33_0_4_20170117201021349.jpg.chip.jpg').resize(image_shape[:2])) / 255
# print(img.shape)
# #先頭に次元を追加
# img = img[np.newaxis]
# print(img.shape)
# #末尾に次元を追加
# img = img[:, :, :, np.newaxis]
# print(img.shape)

# #予測
# predictions_single = model.predict(img)
# print(predictions_single)
# print(predictions_single.shape)
# print(predictions_single[0].argmax())