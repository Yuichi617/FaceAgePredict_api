import json
import urllib.parse
import urllib.request

# 画像データの読み込み
f = open("33_0_4_20170117201021349.jpg.chip.jpg", "rb")
reqbody = f.read()
f.close()
# print(f)
# print(reqbody)

# リクエストの作成
url = "http://127.0.0.1:8080/face-age-predict"
req = urllib.request.Request(
    url,
    data =reqbody,
    method="POST",
    headers={"Content-Type": "application/octet-stream"},
)

# リクエストの送信とレスポンスの受け取り
with urllib.request.urlopen(req) as res:
    print(json.loads(res.read()))