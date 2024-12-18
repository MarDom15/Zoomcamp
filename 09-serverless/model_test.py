import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"
payload = {"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}
response = requests.post(url, json=payload)
print(response.json())
