import requests

ip = requests.get('https://api.ipify.org').text
print(f"My server IP is: {ip}")
