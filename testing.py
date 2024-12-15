# import requests

# url = "http://127.0.0.1:8000/analyze-sentiment/"
# params = {"text": "I love using PyTorch and FastAPI!"}

# response = requests.post(url, params=params)
# print(response.json())

import requests

# URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/predict-images/"

# Path to the image
image_path = "/home/abhilash/Downloads/images.jpeg"

# Send the image
with open(image_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

# Print the response
print(response.json())