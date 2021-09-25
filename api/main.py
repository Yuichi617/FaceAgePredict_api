from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # 全てのオリジンからのアクセスを許可
model = None

# GETテスト用
@app.route('/get_test', methods=["GET"])
def get_test():
    response = {
        "success": True,
        "method": "GET"
    }
    return jsonify(response)

# POSTテスト用
@app.route('/post_test', methods=["POST"])
def post_test():
    print(request.data)
    response = {
        "success": True,
        "method": "POST"
    }
    return jsonify(response)

# 本体
@app.route('/face-age-predict', methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    try:
        # 画像データの前処理
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

        # 予測
        pre_age = int(model.predict(img))
        print("preage: " + str(pre_age))

        # レスポンスデータの作成
        response["prediction"] = pre_age
        response["success"] = True

        return jsonify(response)

    except Exception as e:
        print(e)  # デバッグ用
        return "error"

def Load_Model():
    global model
    print(" * Loading pre-trained model ...")
    model = tf.keras.models.load_model('My_model-opt')
    print(' * Loading end')
    
if __name__ == '__main__':
    Load_Model()
    print(" * Flask starting server...")
    app.run(host='127.0.0.1', port=8080, debug=True)