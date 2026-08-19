import urllib.request
import json

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    '--' + boundary + '\r\n'
    'Content-Disposition: form-data; name="audio"; filename="test.webm"\r\n'
    'Content-Type: audio/webm\r\n\r\n'
    'dummy audio content\r\n'
    '--' + boundary + '--\r\n'
).encode('utf-8')

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/query/voice',
    data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
)

try:
    res = urllib.request.urlopen(req)
    print('Voice Status:', res.status)
    print('Voice Response:', json.loads(res.read().decode('utf-8')))
except urllib.error.HTTPError as e:
    print('HTTPError:', e.code, e.read().decode('utf-8'))
