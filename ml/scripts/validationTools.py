
# LIBRARIES
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython
from keras.models import Sequential, Model, load_model
from keras.utils import to_categorical
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)

#FUNCTIONS
def load_audio_wav(path):
    y, fs = librosa.core.load(path, mono=True, sr=48000)
    y = y[0:132096]
    return np.array(y)

def load_audio_csv(path):
    df = pd.read_csv(path, header = None)
    data = df.values
    y=np.asarray(data[1:,1],np.float32)
    return y

def normalize_audio(audio):
    return audio/peak(audio)

def normalize_spec(spec):
    return (spec+80)/80

def compressor(audio,threshold=None,ratio=1/2):
    if threshold==None: threshold = (1-(peak(audio)-rms(audio)))
    compressedSignal = audio.copy()
    for i in range(len(audio)):
        if audio[i]>threshold:  compressedSignal[i] = +threshold + (np.abs(threshold-audio[i]))*ratio
        if audio[i]<-threshold: compressedSignal[i] = -threshold - (np.abs(-threshold-audio[i]))*ratio
        if compressedSignal[i]> +1.0: compressedSignal[i] = +1.0
        if compressedSignal[i]< -1.0: compressedSignal[i] = -1.0
    print("Threshold set to: ",np.round(threshold,2))
    return compressedSignal

def peak(signal):
    return np.max([np.abs(np.max(signal)), np.abs(np.min(signal))])

def rms(signal):
    return np.sqrt(np.sum(signal**2)/len(signal))

def crest_factor(signal):
    return peak(signal)/rms(signal)

def play_audio(signal):
    print("Samples: ",len(signal))
    print("Duration: ",len(signal)/fs, " ms")
    IPython.display.Audio(data=signal,rate=fs,autoplay=True)

def sawtooth(x,rate=1000):
    return x/rate

def generate_signal(function, full_size=132096,buffer_size=960):
    fullAudio = np.zeros([full_size])
    audiobuffer = np.zeros([buffer_size])
    for x in range(0,buffer_size):
        audiobuffer[x] = function(x=x)
    current_sample = 0
    isrecording = True
    while isrecording == True:
        for i in range(0,buffer_size):
            if current_sample < full_size:
                fullAudio[current_sample] = audiobuffer[i]
            if current_sample > full_size -1:
                isrecording = False
                break
            current_sample += 1
    return fullAudio
    
def load_spectrogram_csv(path):
    df = pd.read_csv(path,header = None)
    data = df.values
    return data

def melspectrogram(audio):
    spectrogram = np.flipud(librosa.power_to_db(librosa.feature.melspectrogram(
                y=audio, sr=48000, n_mels = 128, n_fft=2048, hop_length=1024, fmax=20000), ref=np.max))
    return spectrogram[:,1:-1]

def get_features(audio,normalize=False):
    features = melspectrogram(audio).reshape(1,128,128,1)
    if normalize: features = normalize_spec(features)
    return features

def load_predictions(path):
    df = pd.read_csv(path, header = None)
    data = df.values
    labels = np.asarray(data[1:,1])
    predictions = np.asarray(data[1:,2],dtype = np.float32)
    return labels, predictions

def label_range(spectrogram):
    max_string = 'max:' + str(np.round(np.max(spectrogram),2))
    min_string = 'min:' + str(np.round(np.min(spectrogram),2))
    coor = (0.1)*np.shape(spectrogram)[0]
    plt.text(x=coor*8,y=coor,s=max_string,horizontalalignment='center',color='w',bbox=dict(facecolor='w', alpha=0.2))
    plt.text(x=coor*2,y=coor,s=min_string,horizontalalignment='center',color='w',bbox=dict(facecolor='w', alpha=0.2))
    
def label_diff(spectrogram_diff,vmin=0,vmax=1):
    spectrogram_error = 'e < '+ str(np.round(((np.max(spectrogram_diff)/np.abs(vmax-vmin))*100),2))+' %'
    coor = (0.1)*np.shape(spectrogram_diff)[0]
    plt.text(x=coor*8,y=coor*9,s=spectrogram_error,horizontalalignment='center',color='w',bbox=dict(facecolor='w', alpha=0.2))

def get_spectrograms(python_path,python_type,android_path,android_type, normalize_python=False, normalize_android=False):
    # PYTHON:
    if python_type =='audio_array':
        python_spectrogram = melspectrogram(python_path)
    elif python_type =='audio_wav':
        python_audio = load_audio_wav(python_path)
        python_spectrogram = melspectrogram(python_audio)
    elif python_type =='audio_csv':
        python_audio = load_audio_csv(python_path)
        python_spectrogram = melspectrogram(python_audio)
    elif python_type == 'spec_npy':
        python_spectrogram = np.load(python_path) 
    elif python_type == 'spec_csv':
        print("undefined for python.")  
    # ANDROID:
    if android_type =='audio_array':
        android_spectrogram = melspectrogram(android_path)
    elif android_type =='audio_wav':
        android_audio = load_audio_wav(android_path)
        android_spectrogram = melspectrogram(android_audio)
    elif android_type =='audio_csv':
        android_audio = load_audio_csv(android_path)
        android_spectrogram = melspectrogram(android_audio)
    elif android_type == 'spec_npy':
        android_spectrogram = np.load(android_path) 
    elif android_type == 'spec_csv':
        android_spectrogram = load_spectrogram_csv(android_path)  
    if normalize_python : python_spectrogram=normalize_spec(python_spectrogram)
    if normalize_android : android_spectrogram=normalize_spec(android_spectrogram)
    return python_spectrogram, android_spectrogram

def plot_differences(python_spectrogram, android_spectrogram,figname="new",model=None,labels=None,normalized=False,android_predictions=None):
    if normalized: 
        vmax=1
        vmin=0
    else :
        vmax=0
        vmin=-80              
    # DIFFERENCES:
    differences_spectrogram =np.abs(android_spectrogram - python_spectrogram) 
    # PREDICTIONS
    if model != None :
        if android_predictions == None:
            android_predictions = model.predict(np.array(android_spectrogram.reshape(1,128,128,1)))[0]
            print("android_predictions: empty, using tensorflow model instead.")
        else: android_predictions = load_predictions(android_predictions)[1]
        python_predictions = model.predict(np.array(python_spectrogram.reshape(1,128,128,1)))[0]
        differences_predictions = np.abs(android_predictions-python_predictions)
        prediction_error = np.round((np.max(differences_predictions)*100),2)
        print("Prediction Difference< ",prediction_error,"%")
        rows = 2
        height = 12
        width = 8
    else: 
        rows = 1
        height = 14
        width = 4        
    # PLOT:  
    font = {'weight' : 'normal','size': 13}
    plt.rc('font', **font)
    plt.figure(figsize=(height,width))
    plt.subplot(rows,3,1)
    plt.title('Espectrograma Python')
    plt.imshow(python_spectrogram,vmin=vmin,vmax=vmax,cmap='magma')
    label_range(python_spectrogram)
    plt.colorbar(format='%+2.1f')
    plt.ylabel("Frecuencia [bins]")
    plt.subplot(rows,3,2)
    plt.title('Espectrograma Android')
    plt.imshow(android_spectrogram,vmin=vmin,vmax=vmax,cmap='magma')
    label_range(android_spectrogram)
    plt.ylabel("")
    plt.yticks([])
    plt.colorbar(format='%+2.1f')
    plt.subplot(rows,3,3)
    plt.title('Diferencias')
    plt.imshow(differences_spectrogram,cmap='gist_heat')
    plt.clim(0,np.max(differences_spectrogram))
    label_diff(differences_spectrogram,vmin,vmax)
    plt.colorbar(format='%+2.3f')
    plt.ylabel("")
    plt.yticks([])
    if model!=None:
        plt.subplot(rows,3,4)
        plt.title('TF Python')
        plt.bar(labels,python_predictions,color='orange')
        plt.ylim([0,1])
        plt.ylabel("Activacion")
        plt.xticks(rotation='vertical')
        plt.subplot(rows,3,5)
        plt.title('TFLite Android')
        plt.bar(labels,android_predictions,color='orange')
        plt.ylim([0,1])
        plt.ylabel("")
        plt.xticks(rotation='vertical')
        plt.subplot(rows,3,6)
        plt.title('Diferencias')
        plt.bar(labels,differences_predictions,color='red')
        plt.ylabel("")
        plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig(figname+"_differences",dpi=300,bbox_inches='tight')

def compare_compression(audio_array,makeupGain=1.0,model=None,labels=None):
    threshold = (1-(peak(audio)-rms(audio)))
    compressed_audio = normalize_audio(compressor(audio,threshold))
    plt.figure(figsize=(15,4))
    plt.subplot(131)
    plt.title("Original Signal")
    plt.plot(audio)
    plt.legend(["original"])
    plt.subplot(132)
    plt.title("Compression")
    plt.plot(normalize_audio(audio))
    plt.plot(compressor(audio),'orange')
    plt.axhline(threshold,linestyle="-.",color='r')
    plt.axhline(-threshold,linestyle="-.",color='r')
    plt.legend(["original normalized","compressed"])
    plt.subplot(133)
    plt.title("Normalized")
    plt.plot(compressed_audio,'orange')
    plt.plot(normalize_audio(audio))
    plt.legend(["compression","original"])
    if model!=None and labels != None:
        no_compressed_pred=labels[np.argmax(model.predict(get_features(audio,normalize=True))[0])]
        compressed_pred=labels[np.argmax(model.predict(get_features(compressed_audio,normalize=True))[0])]
        print("Prediction with no Compression: ",no_compressed_pred)
        print("Prediction with Compression: ",compressed_pred)
        
def batch_predict_csv(folder_dir,model,labels,threshold=None, ratio=1/4):
    
    for file in os.listdir(folder_dir):
        print("_____________________________________")
        print("Filename:",file)
        audio_uncompressed = load_audio_csv(os.path.join(folder_dir,file))
        prediction_uncompressed = labels[np.argmax(model.predict(get_features(audio_uncompressed,normalize=True))[0])]
        audio_compressed = normalize_audio(compressor(audio_uncompressed,threshold,ratio))
        prediction_compressed = labels[np.argmax(model.predict(get_features(audio_compressed,normalize=True))[0])]
        print("          Uncompressed | Compressed")
        print("Prediction--->",prediction_uncompressed,"|",prediction_compressed)
        print("CrestFactor------>",np.round(crest_factor(audio_uncompressed),2),"|",np.round(crest_factor(audio_compressed),2))
        print(" ")