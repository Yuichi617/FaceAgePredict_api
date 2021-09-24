import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

#モデルのロード
model = tf.keras.models.load_model("../api/My_model-opt")

# 画像ファイルの読み込み
f = open("33_0_4_20170117201021349.jpg.chip.jpg", "rb")
img = f.read()
f.close()
 # バイナリファイルのデータ(バイト型)を1次元のnumpy配列に変換
_bytes = np.frombuffer(img, dtype=np.uint8)
# バイナリファイルのデータのをデコード (RGB変換)
img2 = cv2.imdecode(_bytes, flags=cv2.IMREAD_COLOR)
# リサイズ
img2 = cv2.resize(img2, dsize=(100, 100))
# データ型の変換＆正規化
img2 = img2.astype('float32') / 255
# #先頭に次元を追加
img2 = img2[np.newaxis]
print(img2.shape)

#予測
predictions_single = model.predict(img2)
print(predictions_single)