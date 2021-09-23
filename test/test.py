import json
import urllib.parse
import urllib.request

# read image data
f = open("MNIST_sample1.png", "rb")
reqbody = f.read()
f.close()

# create request with urllib
url = "http://127.0.0.1:8080/face-age-predict"
req = urllib.request.Request(
    url,
    reqbody,
    method="POST",
    headers={"Content-Type": "application/octet-stream"},
)

# send the request and print response
with urllib.request.urlopen(req) as res:
    print(json.loads(res.read()))