import requests

url = "http://localhost:5000/detect"
image_path = "../imgs/img1.jpg"

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

print(response.json())