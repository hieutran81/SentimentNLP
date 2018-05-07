import requests
r = requests.post('http://0.0.0.0:5000/sentiment', data={'text': 'hdmi mở ra một kỷ nguyên mới'})
print(r.text)