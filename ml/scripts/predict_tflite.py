import os
import csv
import time
import scipy
import shutil
import random
import sklearn
import datetime
import librosa
import librosa.display
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import keras
import argparse

def load_labels(labels_path):
    with open(labels_path) as file:
        labels = [line.rstrip('\n') for line in file.readlines()]
    return labels

def load_audio(path, sr=48000, mono=True, audio_format="wav"):
    if audio_format == "wav":
        audio,fs = librosa.load(path,sr,mono=mono)
    elif audio_format == "csv":
        df = pd.read_csv(path, header = None)
        audio=np.asarray(df.values[1:,1],np.float32)
    return audio

def split_audio_clips(audio,clip_length):
    n_clips=int(np.floor(len(audio)/clip_length))
    audio=audio[0:(n_clips*clip_length)]
    clips = np.split(audio,n_clips)
    return clips

def melspectrogram(audio,ref=np.max,sr=48000,n_mels=128,n_fft=2048,hop_length=1024,fmax=20000):
    S = librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=n_mels,n_fft=n_fft,hop_length=hop_length,fmax=fmax)
    spectrogram = np.flipud(librosa.power_to_db(S, ref=ref))  
    return spectrogram[:,1:-1]

def map_norm(data,data_min,data_max):
    if data_min < 0:
        data+=np.abs(data_min)
        data_max+=np.abs(data_min)
    elif data_min >= 0 :
        data-=np.abs(data_min)
        data_max-=np.abs(data_min)
    data = data/np.abs(data_max)
    return data

def tflite_inference(input_data,model_path): 
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def tflite_predict_audio(audio_path,model_path,labels_path,show_im=True):
    labels = load_labels(labels_path)
    y=load_audio(audio_path)
    full_spec = melspectrogram(y)
    clips = split_audio_clips(y,clip_length=132096)
    audio_predictions=np.zeros([len(labels)])
    predictiongram = []
    for clip in clips:
        feature = melspectrogram(clip)
        feature = np.reshape(feature,newshape=(1,128,128,1))
        feature = map_norm(data=feature,data_min=-80,data_max=0)
        out = tflite_inference(input_data=feature,model_path=model_path)[0]
        #print(out)
        audio_predictions += keras.utils.to_categorical(np.argmax(out),num_classes=len(labels))
        predictiongram.append(out)
    total_samples = int(np.sum(audio_predictions))
    gs = gridspec.GridSpec(nrows= 2, ncols= 2,wspace=0.5)
    fig = plt.figure(figsize=(10,8))
    plt.subplot(gs[0,0])
    plt.plot(y,color='darkblue')
    plt.xlim([0,len(y)])
    plt.xticks([])
    plt.title(audio_path)
    plt.xlabel("Time")
    plt.ylabel("Amplitud")
    plt.subplot(gs[0,1])
    plt.barh(labels,audio_predictions,color='darkblue',align='center')
    plt.xlim([0,np.sum(audio_predictions)])
    plt.title("Predicciones")
    plt.subplot(gs[1,0]) 
    im = plt.imshow(np.flipud(np.transpose(predictiongram)),cmap='YlGnBu',interpolation='hanning',aspect='auto')
    plt.title("Predictograma")
    plt.xlabel("Frames")
    plt.yticks(np.arange(len(labels)),labels[::-1])
    plt.colorbar(im,shrink=1,panchor='false',orientation="horizontal",spacing='uniform',pad=0.2)
    plt.subplot(gs[1,1])
    im = plt.imshow(full_spec,cmap='YlGnBu',interpolation='hanning',aspect='auto')
    plt.title("Espectrograma")
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0,total_samples)
    plt.xlabel("Frames")
    plt.ylabel("Mels")
    fig_name = "BA_eval_"+audio_path+".png"
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')
    if show_im: plt.show()
    fig.clear()
    plt.close(fig)
    return audio_predictions

# main execution:
parser = argparse.ArgumentParser(description="Predict Audio Scene using TFLite model")
parser.add_argument("audio_path",type=str,help="Path to audio file for aplying scene classifier")
args = parser.parse_args()
tflite_predict_audio(audio_path=args.audio_path,model_path,labels_path)