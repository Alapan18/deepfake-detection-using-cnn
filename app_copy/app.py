#imports
from flask import Flask,request, url_for, redirect, render_template
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
from torch.autograd import Variable
import sys
import random
from torch import nn
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights
import glob
import face_recognition
from audio_extract import extract_audio
import shutil
import os
from flask import Flask, request, render_template
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from moviepy.editor import VideoFileClip
#imports


app = Flask(__name__)

#Model architecture declaration
class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))

#image to tensor    
im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png',image*255)
    return image


predictions = []
confidences = []
#prediction
def predict(model, img, path='./'):
    # Ensure the model is in evaluation mode and the input is on the correct device
    model.eval()

    if not isinstance(img, torch.Tensor) or img.shape[0] == 0:
        print("Warning: Empty or invalid input image tensor.")
        return None

    # Ensure the input is on the correct device
    img = img.to('cuda')

    # Loop through each frame in the batch
    for i in range(img.size(1)):  # img.size(1) gives the number of frames
        frame = img[:, i, :, :, :].unsqueeze(0)  # Extract each frame and add batch dimension

        # Forward pass
        fmap, logits = model(frame)

        # Extract weights from the final linear layer
        try:
            weight_softmax = model.linear1.weight.detach().cpu().numpy()
        except AttributeError:
            print("Error: Model does not have a layer named 'linear1'.")
            return None

        # Softmax and prediction
        logits = torch.nn.functional.softmax(logits, dim=1)  # Ensure softmax is applied
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100

        # Initialize lists to store predictions and confidence values for each frame


        # Store results
        if int(prediction.item()) == 1:
            predictions.append(1)
        else:
            predictions.append(0)
        confidences.append(confidence)
        #print(predictions)

        print(f'Frame {i + 1}: Prediction = {prediction.item()}, Confidence = {confidence:.2f}%')

    # Optionally, return the list of predictions and confidences
    return [predictions, confidences]

#frame extraction
class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length = 60,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        # Extract all frames from the video
        for frame in self.frame_extract(video_path):
            frames.append(self.transform(frame))
        # Stack all frames into a tensor
        frames = torch.stack(frames)
        return frames.unsqueeze(0)


    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

#image transform
train_transforms = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize((im_size,im_size)),
  transforms.ToTensor(),
  transforms.Normalize(mean,std)])


#model load
model = Model(2).cuda()
path_to_model = 'C:/Users/alapa/Documents/Deepfake_detection_using_deep_learning-master/Deepfake_detection_using_deep_learning-master/train_model_97.49_epoch_40.pt'
model.load_state_dict(torch.load(path_to_model))
model.eval()

#percentage calculation
def percent(a):
    frames=len(a)
    count=0
    for k in range(0,frames):
        if a[k] == 1:  # Assuming 1 means 'real'
                count = count + 1
    percentt=(count/frames)*100
    print(frames, " ",count)
    return percentt

#extract audio
def audioslice(a,b):
    c="./audio/"+b+".wav"
    video = VideoFileClip(a)
    video.audio.write_audiofile(c, codec='pcm_s16le')
    return c
#analyze audio
def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def analyze_audio(input_audio_path):
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"
    #DEBUGGING CODES
    if not os.path.exists(input_audio_path):
        return "Error: The specified file does not exist."
    elif not input_audio_path.lower().endswith(".wav"):
        return "Error: The specified file is not a .wav file."

    mfcc_features = extract_mfcc_features(input_audio_path)
    if mfcc_features is not None:
        scaler = joblib.load(scaler_filename)
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))

        svm_classifier = joblib.load(model_filename)
        prediction = svm_classifier.predict(mfcc_features_scaled)

        if prediction[0] == 0:
            return " REAL"
        else:
            return " FAKE"
    else:
        return "Unable to process the input audio."
#clear global lists
def clear():
    predictions.clear()
    confidences.clear()

#clear previous video and audio
def clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate through the files and subdirectories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it's a file or directory and remove accordingly
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and all its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {folder_path} does not exist.')

@app.route('/', methods=['GET', 'POST'])
def handle_predict():
    if request.method == "POST":
        if "video" not in request.files:
            return render_template("web.html", prediction='Please Choose a video')
        video=request.files['video']
        if video.filename == "":
            return render_template("web.html", prediction='Please Choose a video')
        if video:
            path_of_videos="C:/Users/alapa/Documents/Deepfake_detection_using_deep_learning-master/Deepfake_detection_using_deep_learning-master/app_copy/video/"+str(video.filename)
            b=str(video.filename)
            b=b[:-4]
            video.save(path_of_videos)
            
            path_to_videos=[path_of_videos]
            video_dataset = validation_dataset(path_to_videos,sequence_length = 20,transform = train_transforms)
            # Initialize sum to accumulate confidence values
            sum_confidence = 0
            
            # Process the single video
            p = predict(model, video_dataset[0], './')  # Assuming there's only one video in video_dataset
            print("Predicting audio")
            result = analyze_audio(audioslice(path_of_videos, b))
            print("Audio prediction done")
            if p is not None:
                predictions = p[0]
                confidences = p[1]
                #count=0
            # frame_no=0
                # Accumulate confidence values based on predictions
                for i in range(len(predictions)):
                #  frame_no = frame_no + 1
                    if predictions[i] == 1:  # Assuming 1 means 'real'
                    # count = count + 1
        
                        sum_confidence += confidences[i]
                    else:  # Assuming 0 means 'fake'
                        sum_confidence += 100 - confidences[i]
                #print(predictions)
                percentage=percent(predictions)
                # Calculate the average confidence
                avg_confidence = sum_confidence / len(predictions)

                #print(frame_no, " ",count)
                if avg_confidence < 50:
                    fake_confidence = 100 - avg_confidence
                    #p1 = f"Fake | Confidence: {fake_confidence:.2f}"
                    p1 = f"Fake | Confidence: {fake_confidence:.2f} | Percentage of reality:{percentage:.2f} | Audio Detection Result:"+str(result)
                    clear()
                    clear_folder('./video')
                    clear_folder('./audio')
                    return render_template("web.html", prediction=p1)
                else:
                    #p1 = f"Real | Confidence: {avg_confidence:.2f}"
                    p1 = f"Real | Confidence: {avg_confidence:.2f} | Percentage of reality:{percentage:.2f} | Audio Detection Result:"+str(result)
                    clear()
                    clear_folder('./video')
                    clear_folder('./audio')
                    return render_template("web.html", prediction=p1)
                
            else:
                return render_template("web.html", prediction='Prediction failed for the video.')
    return render_template("web.html")

    


if __name__ == '__main__':
    app.run(debug=True)