from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.models import resnext50_32x4d
from torchvision.models import ResNeXt50_32X4D_Weights
import cv2
import numpy as np
import os

app = Flask(__name__)

# Define the model class
class Model(torch.nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(model.children())[:-2])
        self.lstm = torch.nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = torch.nn.LeakyReLU()
        self.dp = torch.nn.Dropout(0.4)
        self.linear1 = torch.nn.Linear(2048, num_classes)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Initialize the model
model = Model(num_classes=2)
model.load_state_dict(torch.load('C:/Users/alapa/Documents/Deepfake_detection_using_deep_learning-master/Deepfake_detection_using_deep_learning-master/train_model_97.49_epoch_40.pt'))
model.eval()

# Define dataset class
class ValidationDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform

    def __len__(self):
        return 1  # Only one video per dataset

    def __getitem__(self, idx):
        frames = []
        for frame in self.frame_extract(self.video_path):
            frames.append(self.transform(frame))
        frames = torch.stack(frames)
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success, image = vidObj.read()
        while success:
            yield image
            success, image = vidObj.read()

# Define image transformations
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def predict(video_path):
    dataset = ValidationDataset(video_path, transform=train_transforms)
    video_frames = dataset[0]
    model.eval()
    predictions = []
    confidences = []
    for i in range(video_frames.size(1)):
        frame = video_frames[:, i, :, :, :].unsqueeze(0)
        fmap, logits = model(frame)
        logits = torch.nn.functional.softmax(logits, dim=1)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
        predictions.append(prediction.item())
        confidences.append(confidence)
    sum_confidence = sum(confidences)
    avg_confidence = sum_confidence / len(predictions)
    return "Real" if avg_confidence >= 50 else "Fake", avg_confidence

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        video_path = 'uploaded_video.mp4'
        file.save(video_path)
        label, confidence = predict(video_path)
        os.remove(video_path)
        return jsonify({"label": label, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
