from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
# from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# app.add_middleware(
#     # CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins (you can restrict this in production)
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all methods
#     allow_headers=["*"],  # Allow all headers
# )

# Load a pretrained model for image classification
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Load ImageNet class labels
imagenet_classes = []
with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
async def read_root():
    return {"message": "FastAPI GPU Server Running!"}

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    # Open the uploaded image
    image = Image.open(file.file).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        prediction_idx = probabilities.argmax().item()

    # Get the class label and confidence
    class_name = imagenet_classes[prediction_idx]
    confidence = round(probabilities[prediction_idx].item(), 4)

    return {"class_name": class_name, "confidence": confidence}

@app.post("/predict-images/")
async def predict_image(file: UploadFile = File(...)):
    # Open the uploaded image
    image = Image.open(file.file).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top 5 predictions
        top5_probabilities, top5_indices = torch.topk(probabilities, 5)

    # Map indices to class names and confidence scores
    top5_results = [
        {"class_name": imagenet_classes[idx], "confidence": round(prob.item(), 4)} 
        for idx, prob in zip(top5_indices, top5_probabilities)
    ]

    return {"top_5_predictions": top5_results}




# from fastapi import FastAPI
# import torch
# from torch.nn import functional as F
# from transformers import BertTokenizer, BertForSequenceClassification

# app = FastAPI()

# # Load pre-trained sentiment model (BERT fine-tuned on sentiment dataset)
# tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
# model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
# model.eval()

# @app.get("/")
# async def read_root():
#     return {"message": "FastAPI GPU Server Running!"}

# @app.post("/analyze-sentiment/")
# async def analyze_sentiment(text: str):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512) 
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = F.softmax(outputs.logits, dim=-1)
#     sentiment = torch.argmax(probs, dim=-1).item()
#     labels = ["very negative", "negative", "neutral", "positive", "very positive"]
#     return {"sentiment": labels[sentiment], "confidence": round(probs[0][sentiment].item(), 4)}





# from fastapi import FastAPI
# import torch
# import torch.nn as nn

# app = FastAPI()

# # Example LSTM Anomaly Detection Model
# class AnomalyDetector(nn.Module):
#     def __init__(self):
#         super(AnomalyDetector, self).__init__()
#         self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2, batch_first=True)
#         self.fc = nn.Linear(128, 1)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         return self.fc(lstm_out[:, -1])

# model = AnomalyDetector()
# model.load_state_dict(torch.load("anomaly_detector.pth"))
# model.eval()

# @app.post("/detect-anomaly")
# async def detect_anomaly(data: list[float]):
#     input_tensor = torch.tensor(data).unsqueeze(0).unsqueeze(-1)
#     with torch.no_grad():
#         prediction = model(input_tensor)
#     is_anomaly = prediction.item() > 0.5
#     return {"is_anomaly": is_anomaly, "score": prediction.item()}
