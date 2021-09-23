from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)
model = None

# def Load_Model():
#     global model
#     print(" * Loading pre-trained model ...")
#     model = tf.keras.models.load_model('My_model-opt')
#     print(' * Loading end')

@app.route('/face-age-predict', methods=["POST"])
def predict():
    try:
        # print(request.data)
        # バイナリファイルのデータ(バイト型)を1次元のnumpy配列に変換
        _bytes = np.frombuffer(request.data, dtype=np.uint8)
        # バイナリファイルのデータのをデコード (RGB変換)
        img = cv2.imdecode(_bytes, flags=cv2.IMREAD_COLOR)
        # リサイズ
        img = cv2.resize(img, dsize=(100, 100))
        # データ型の変換＆正規化
        img = img.astype('float32') / 255
        # #先頭に次元を追加
        img = img[np.newaxis]

        return jsonify(img.shape), 200

    except Exception as e:
        print(e)  # デバッグ用
        return "error"

if __name__ == '__main__':
    # Load_Model()
    print(" * Flask starting server...")
    app.run(host='127.0.0.1', port=8080, debug=True)