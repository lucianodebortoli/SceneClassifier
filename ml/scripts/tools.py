# Version: 2019.07.25
# Author: Luciano De Bortoli

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
from contextlib import redirect_stdout
from tensorflow import lite # needs nightly build for using TFconverter
import keras
print ("Keras Version: ", keras.__version__)

def save_text(string,savemane):
    text_file = open(savemane+".txt","w")
    text_file.write(string)
    text_file.close()

def list_print(x,y,spacing=40):
    print(x,"-"*(spacing-len(x)),y)

def random_RGB(n_colors=1):
    colors = np.random.uniform(low=0.25, high=0.75,size=(n_colors,3))
    return colors

def progress(element,parent_list,subelement=None,n_bars=20):
    ratio = np.round(parent_list.index(element)/len(parent_list),3)
    n_completed = int(np.floor(ratio*n_bars))
    n_remaining = n_bars - n_completed
    percent = int(round(ratio,2)*100)
    if subelement==None: subelement=element
    print("█"*n_completed," "*n_remaining,"|",percent,"%","►", subelement," "*40,end='\r')   
    
def uncategorical(array):
    uncat = []
    for element in array:
        uncat.append(np.argmax(element))
    return uncat

def map_norm(data,data_min,data_max):
    if data_min < 0:
        data+=np.abs(data_min)
        data_max+=np.abs(data_min)
    elif data_min >= 0 :
        data-=np.abs(data_min)
        data_max-=np.abs(data_min)
    data = data/np.abs(data_max)
    return data

def get_size_support(main_dir):
    folders = os.listdir(main_dir)
    file_sizes = []
    for folder in folders:
        files = os.listdir(os.path.join(main_dir,folder))
        to_sum = []
        for file in files:
            to_sum.append(os.stat(os.path.join(main_dir,folder,file)).st_size)
        file_sizes.append(np.sum(to_sum))
    return folders, file_sizes

def convert_wavsize_to_seconds(wav_sizes, sample_rate=44100,bit_depth=16):
    bytes_per_second = int((sample_rate * bit_depth) / 8)
    return np.around(np.array(wav_sizes)/ bytes_per_second, decimals=0)

def time_format(time_in_sec,sep = ':'):
    time_format = "%H"+sep+"%M"+sep+"%S"
    return time.strftime(time_format, time.gmtime(time_in_sec))

def time_stamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def print_duration_support(directory,sample_rate=44100,bit_depth=16):
    file_names, file_sizes = get_size_support(directory) 
    for index, file_size in enumerate(file_sizes):
        file_duration = convert_wavsize_to_seconds(file_size)    
        list_print(file_names[index], time_format(file_duration))  

def duration_support(main_dir):
    file_names, class_support = get_size_support(main_dir)
    class_support = convert_wavsize_to_seconds(class_support)
    plt.figure(figsize=(15,3))
    plt.bar(file_names,class_support,color=random_RGB())
    plt.xticks(file_names, file_names,rotation=90)
    plt.title('Support')
    plt.ylabel("Total support [s]")
    fontdict = dict(horizontalalignment='center',verticalalignment='top',color='w')
    for index,support in enumerate(class_support):
        plt.text(index,support,str(int(support))+" s",fontdict=fontdict)
    plt.savefig("Size Support.png",dpi=300,bbox_inches='tight')
    print("min:",time_format(np.min(class_support)))
    print("avg:",time_format(np.mean(class_support)))
    print("max:",time_format(np.max(class_support)))

def load_audio(path, sr=48000, mono=True, audio_format="wav"):
    if audio_format == "wav":
        audio,fs = librosa.load(path,sr,mono=mono)
    elif audio_format == "csv":
        df = pd.read_csv(path, header = None)
        audio=np.asarray(df.values[1:,1],np.float32)
    return audio          

def save_audio(audio,path,sr=48000,norm=False,audio_format="wav"):
    if audio_format == "wav":
        librosa.output.write_wav(path, audio, sr=sr, norm=norm)
    if audio_format == "csv":
        with open(path, mode='w') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for sample_index,sample_value in enumerate(audio):
                file_writer.writerow([sample_index,sample_value])

def split_audio_clips(audio,clip_length):
    n_clips=int(np.floor(len(audio)/clip_length))
    audio=audio[0:(n_clips*clip_length)]
    clips = np.split(audio,n_clips)
    return clips

def merge_wav_clips(data_dir,new_data_dir,sr=48000,min_clip_len_in_sec=10,max_out_len_in_sec=3600):
    folders = os.listdir(data_dir)    
    if not os.path.isdir(new_data_dir): os.mkdir(new_data_dir)
    for folder in folders:
        full_audio = []
        for file in os.listdir(os.path.join(data_dir,folder)):
            progress(folder,folders,subelement=file)
            clip = load_audio(os.path.join(data_dir,folder,file))
            full_audio.append(np.array(clip[0:sr*min_clip_len_in_sec-1],dtype=np.dtype(float)))
        full_audio= np.array(full_audio).flatten() 
        n_splits = int(np.round(len(full_audio)/(sr*max_out_len_in_sec),0))
        audio_splits = np.split(full_audio,n_splits)
        output_dir = os.path.join(new_data_dir,folder)
        os.mkdir(output_dir)
        for split_index, audio_split in enumerate(audio_splits):
            audio_name=folder +"_"+str(split_index)+".wav"
            save_audio(audio_split,os.path.join(output_dir,audio_name))
    print("Finished"," "*60,end="\r")
    
def stft(audio):
    return librosa.core.stft(y=audio, n_fft=2048, hop_length=1024, window='hann', center=True)

def zero_crossing_rate(audio):
    return np.mean(librosa.feature.zero_crossing_rate(y=audio, frame_length=2048, hop_length=1024, center=True),axis=1)

def spectrum(audio):
    return np.abs(stft(audio))
   
def spectral_centroid(S):
    return np.mean(librosa.feature.spectral_centroid(S=S,sr=48000, n_fft=2048, hop_length=1024),axis=1)

def spectral_bandwidth(S):
    return np.mean(librosa.feature.spectral_bandwidth(S=S,sr=48000, n_fft=2048, hop_length=1024, p=2),axis=1)

def spectral_contrast(S):
    return np.mean(librosa.feature.spectral_contrast(S=S,sr=48000, n_fft=2048, hop_length=1024,n_bands=6),axis=1)

def spectral_flatness(S):
    return np.mean(librosa.feature.spectral_flatness(S=S, n_fft=2048, hop_length=1024, amin=1e-10, power=2.0),axis=1)

def spectral_rolloff(S):
    return np.mean(librosa.feature.spectral_rolloff(S=S,sr=48000, n_fft=2048, hop_length=1024),axis=1)

def rms(S):
    return np.mean(librosa.feature.rmse(S=S,frame_length=2048,hop_length=1024))

def mel_spectrogram(S,mels=128):
    return librosa.feature.melspectrogram(S=S,sr=48000,n_mels=mels,n_fft=2048,hop_length=1024,fmax=20000)

def melspectrum(S):
    return np.mean(mel_spectrogram(S),axis=1)

def mfcc(S,mels=128):
    return np.mean(librosa.feature.mfcc(S=mel_spectrogram(S,mels=mels),sr=48000,n_mfcc=128,dct_type=2,norm='ortho'),axis=1)

def spectrogram(audio,ref=np.max):
    S = spectrum(audio)
    spectrogram = np.flipud(librosa.power_to_db(S, ref=ref))  
    return spectrogram[:,1:-1]

def melspectrogram(audio,ref=np.max,sr=48000,n_mels=128,n_fft=2048,hop_length=1024,fmax=20000):
    S = librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=n_mels,n_fft=n_fft,hop_length=hop_length,fmax=fmax)
    spectrogram = np.flipud(librosa.power_to_db(S, ref=ref))  
    return spectrogram[:,1:-1]

def plot_specgram(audio,sr=48000):
    spectrum, freqs, t, im = plt.specgram(x= audio,NFFT=2048,Fs=sr,scale='dB',cmap='magma',mode='magnitude',sides='onesided')  

def specshow(S,title="spectrogram",dB=True):
    librosa.display.specshow(np.flipud(S),y_axis='mel',x_axis='time', fmax=20000)
    if dB: plt.colorbar(format='%+2.0f dB')
    else : plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title,dpi=300,bbox_inches='tight')    
    
def feature_extract(clip,feature_type):
    if feature_type   == 'lin_specs' : return spectrogram(clip)
    elif feature_type == 'lin_means' : return np.mean(spectrogram(clip),axis=1)
    elif feature_type == 'mel_specs' : return melspectrogram(clip)
    elif feature_type == 'mel_means' : return np.mean(melspectrogram(clip),axis=1)
    else: print(feature_type, "Undefined!")
        
def feature_reshape(feature,feature_type):
    if feature_type == 'mel_specs': return np.reshape(feature,newshape=(1,128,128,1))
    elif feature_type == 'lin_specs': return np.reshape(feature,newshape=(1,1025,128,1))
    elif feature_type == 'mel_means': return np.reshape(feature,newshape=(1,128))
    elif feature_type == 'lin_means': return np.reshape(feature,newshape=(1,1025))
    
def save_clips_features(clips, feature_type, save_dir, parent_name):
    for count, clip in enumerate(clips):
        features = feature_extract(clip,feature_type)  
        np.save(file=os.path.join(save_dir, parent_name+"_"+str(count)+".npy"), arr=features)

def dataset_feature_extraction(source_dir,destination_dir,feature_type):
    if os.path.exists(destination_dir):
        print("destination_dir already exists")
        return
    os.mkdir(destination_dir)
    folders_names = os.listdir(source_dir)
    for label, folder in enumerate(folders_names):
        os.mkdir(os.path.join(destination_dir,folder))                                        
        folder_path = os.path.join(source_dir,folder)                   
        audio_files = os.listdir(folder_path)                                                               
        for audio_file in audio_files: 
            progress(folder,folders_names,subelement=audio_file)
            audio = load_audio(os.path.join(folder_path,audio_file))                
            clips = split_audio_clips(audio,clip_length=132096)
            save_dir=os.path.join(destination_dir,folders_names[label])
            save_clips_features(clips,feature_type,save_dir=save_dir,parent_name=audio_file)           
    print("Dataset Built"+" "*50) 
    
def plot_pie(sizes): 
    pie_parameters = {
        'x': sizes,
        'labels':["TRAIN","VAL","TEST"],
        'colors':['lightblue',"lightgreen","gold" ],
        'autopct':'%1.1f%%',
        'startangle':45,
        'explode': [0]*len(sizes)
    }
    fig1, ax1 = plt.subplots()
    ax1.pie(**pie_parameters)
    centre_circle = plt.Circle((0,0),0.85,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax1.axis('equal') 
    plt.title("DATA SPLIT",fontsize='large',fontweight='bold')
    plt.xlabel('Total Data Samples:'+str(np.sum(sizes)))
    plt.show()
    
def print_all_spectrograms(dataset_dir,new_dataset_dir,fig_save_dir):
    os.mkdir(fig_save_dir)
    folders = os.listdir(new_dataset_dir)    
    for folder in folders:
        progress(folder,folders)
        files = os.listdir(os.path.join(new_dataset_dir,folder))
        for file in files:
            audio,fs = load_audio(os.path.join(new_dataset_dir,folder,file))
            plt.figure(figsize=(15,3))
            specshow(np.flipud(melspectrogram(audio)),title("file"))
    print("Finished"," "*40,end="\r")
    
def visualize_mapping(evaluation_labels,model_labels):
    print("Evaluation Folders ------- >  Model Labels ")
    for i in range(len(evaluation_labels)):
        print(evaluation_labels[i], ("-"*(25-len(evaluation_labels[i]))),">",model_labels[i])
        
def get_class_weights(Y_data):
    class_weights = sklearn.utils.class_weight.compute_class_weight(
    'balanced',np.unique(uncategorical(Y_data)),uncategorical(Y_data))
    return class_weights   

def spec_mask(data,low_bin,high_bin,null=0,flip=False):
    if flip : data = np.array([np.flipud(spec) for spec in data])
    count,bin_size,frame_size,n_channels=np.shape(data)
    data[:,0:low_bin,:,:] = null
    data[:,high_bin:bin_size,:,:] = null
    if flip : data = np.array([np.flipud(spec) for spec in data])
    return data
    
def spec_dropout(X_batch,freq_strips=True,time_strips=True,n_strips=1,min_strip_size=1,tolerance=0.2,null_value=0):
    batch_size, bin_size, frame_size, n_channels = np.shape(X_batch)
    n_samples_to_augment = np.random.randint(low=int(0.25*batch_size),high=int(0.75*batch_size))
    augment_selection = random.sample(list(np.arange(batch_size)),n_samples_to_augment)    
    limit = np.minimum(bin_size,frame_size)
    for _ in range(n_strips):
        strip_sizes = list(np.random.randint(low=min_strip_size,high=0.2*limit/2,size=n_samples_to_augment))
        strip_centers = [int(np.random.uniform(strip_size,limit-strip_size)) for strip_size in strip_sizes]
        strips_selection = [list(np.arange(center-size,center+size+1)) for size, center in zip(strip_sizes,strip_centers)]       
        for i in range(n_samples_to_augment):
            if freq_strips: X_batch[augment_selection[i],strips_selection[i]] = null_value
            if time_strips: X_batch[augment_selection[i],:,strips_selection[i]] = null_value      
    return X_batch

def spec_filters(X_batch):
    batch_size, bin_size, frame_size, n_channels = np.shape(X_batch) 
    X_batch = np.reshape(X_batch,newshape=(batch_size,bin_size,frame_size))
    batch_filters = scale_filters(butterworth_filters(bin_size=bin_size,n_filters=batch_size),n_mels=bin_size)
    batch_matrices = [np.tile(batch_filter,(bin_size,1)).transpose() for batch_filter in batch_filters]
    filtered_batch = [x*np.flipud(matrix) for x,matrix in zip(X_batch,batch_matrices)]
    return np.reshape(filtered_batch,newshape=(batch_size, bin_size, frame_size,n_channels)) 

def butterworth_filters(bin_size,n_filters=1,wc_lims=[0.4,1],n_lims=[1,20]):
    w_norm = np.linspace(0,1,bin_size)
    butterworth_filters = []
    while n_filters>0:
        wc = np.random.uniform(*wc_lims)
        n  = np.random.randint(*n_lims)       
        butterworth_lpf = [1/np.sqrt(1+(w/wc)**(2*n)) for w in w_norm]
        butterworth_hpf = np.flip(butterworth_lpf)
        butterworth_bpf = butterworth_lpf * butterworth_hpf
        butterworth_filters.append(butterworth_bpf)
        n_filters-=1
    return butterworth_filters

def scale_filters(filter_arrays,n_mels,f_max=20000):
    mel_freqs = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=f_max, htk=False)
    mel_indeces = [int(np.floor(freq*n_mels/f_max)) for freq in mel_freqs]
    filtered_arrays = []
    for filter_array in filter_arrays:
        filtered_array = np.flip([filter_array[index] for index in mel_indeces]) # mel_scale
        filtered_array = scipy.signal.savgol_filter(filtered_array,window_length=9,polyorder=3) # smooth
        while np.any(filtered_array <=0): filtered_array += 0.01
        filtered_array = (20 * np.log10(filtered_array) + 80)/80 # convert to dB & normalize
        filtered_arrays.append(filtered_array)
    return filtered_arrays
    
class Combination:
    def __init__(self, source_path, destination_path):   
        self.source_path = source_path
        self.destination_path = destination_path
        self.class_names = []
        self.dependencies = []
        self.initialize_labels()
        
    def initialize_labels(self):        
        self.sources_labels = []
        labels = os.listdir(self.source_path)
        for label in labels:
            self.sources_labels.append(label)
        
    def display_sources_labels(self):
        for source in self.sources_labels:
            print(source)     

    def display_dependencies(self):
        for class_name, dependence in zip(self.class_names,self.dependencies):
            print(class_name,"-"*(30-len(class_name)), dependence)  
        
    def add(self, class_name, sources_to_add):
        valid = True
        for source in sources_to_add:
            if source not in os.listdir(self.source_path):
                valid = False
                print(source, "not found")
                return
        if valid: 
            self.class_names.append(class_name)
            self.dependencies.append(sources_to_add) 
    
    def display_support(self):
        file_names, file_sizes = get_size_support(self.source_path)
        file_sizes = np.array(file_sizes)/1024/1024
        support=[]
        for dependence in self.dependencies:
            to_sum=[]
            for file_name,file_size in zip(file_names,file_sizes):
                if file_name in dependence: to_sum.append(file_size)
            support.append(np.sum(np.array(to_sum)))  
        plt.figure(figsize=(15,3))
        plt.bar(self.class_names,support,color='black')  
        plt.xticks(self.class_names, self.class_names, rotation=90)
        plt.title("Preliminary Support")
        plt.xlabel("Clases")
        plt.ylabel("Support [MB]")
        print("Total Size:",np.round(np.sum(support),2),"MB")
    
    def generate(self):
        if os.path.exists(self.destination_path):
            print("destination_path already exists")
            return
        os.mkdir(self.destination_path)
        for label in self.class_names:
            progress(label,self.class_names)
            os.mkdir(os.path.join(self.destination_path,label))
            current_dependencies = self.dependencies[self.class_names.index(label)]
            current_dependencies = list(np.unique(np.array(current_dependencies)))
            for dependence in current_dependencies:
                for folder in self.sources_labels:
                    if dependence == folder:
                        source_files = os.listdir(os.path.join(self.source_path,folder))
                        for file in source_files:
                            source_dir = os.path.join(self.source_path,folder,file)
                            dest_dir   = os.path.join(self.destination_path,label,file)
                            shutil.copyfile(source_dir,dest_dir)
        print("sucessfully generated!"," "*40)

class Dataset:
    def __init__ (self,name="data",features_dir=None,xy_dir=None, labels=None):
        self.name         = name
        self.features_dir = features_dir
        self.xy_dir       = xy_dir
        if labels==None   : self.labels= os.listdir(features_dir)
        else              : self.labels= labels
        self.num_classes  = len(self.labels)
        print(self.name,"created!")
        
    def get_file_count(self):
        support=[]
        for label in self.labels:
            files = os.listdir(os.path.join(self.features_dir,label))
            support.append(len(files))
        return support
    
    def plot_support(self,fig_name="support"):
        plt.figure(figsize=(15,3))
        support = self.get_file_count()
        plt.bar(self.labels,support,color=random_RGB())  
        plt.xticks(self.labels, self.labels,rotation=90)
        plt.title(fig_name)
        plt.xlabel("Clases")
        plt.ylabel("Support [files]")
        plt.savefig(fig_name+".png",dpi=300,bbox_inches='tight')
        
    def purge(self,n_cycles=1,excess_threshold=0):
        while n_cycles > 0: 
            print("Processing..",end='\r')
            support = self.get_file_count()
            support_avg = int(np.round(np.mean(support),0))
            support_limit = int((excess_threshold+1)*support_avg)   
            print("Dataset Unbalance:",np.min(support),"-",np.max(support))
            print("New Uniform ", int(np.min(support)), "-",support_limit )
            dataset_folders=os.listdir(self.features_dir)
            for folder in dataset_folders:
                progress(folder,dataset_folders)
                files = os.listdir(os.path.join(self.features_dir,folder))
                excess = len(files) - support_limit
                while excess > 0 :
                    random_index = random.randint(0,len(files)-1)
                    os.remove(os.path.join(self.features_dir,folder,files[random_index]))
                    files = os.listdir(os.path.join(self.features_dir,folder))
                    excess = len(files) - support_limit
            print("Finished"," "*40,end='\r') 
            n_cycles-=1
        
    def compile_XY(self,split=False,val_size=0.2,test_size=0.2):
        X  = []
        Y  = []
        num_classes = len(self.labels)
        print("Total Classes:  ",num_classes)
        for folder in self.labels:                                           
            folder_path = os.path.join(self.features_dir,folder)                
            numpy_files = os.listdir(folder_path)                                     
            label = self.labels.index(folder)                                
            progress(folder,self.labels)
            for numpy_file in numpy_files:                                          
                numpy_path = os.path.join(folder_path,numpy_file)                    
                numpy_array = np.load(file=numpy_path)                    
                X.append(numpy_array)
                Y.append(label)
        print("Reshaping Data"," "*40,end='\r')
        X = np.array(X)
        Y = np.array(Y)
        shape = list(np.shape(X))
        shape.append(1) # add channel dimension
        X = np.reshape(X,newshape=tuple(shape))
        Y = keras.utils.to_categorical(Y,num_classes=num_classes)    
        if split:
            print("Saving XY splits"," "*30, end = '\r')
            X_set, X_test, Y_set, Y_test = sklearn.model_selection.train_test_split(X,Y,test_size=test_size)
            val_size = val_size/(1-test_size)
            X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_set,Y_set,test_size=val_size)
            np.save(os.path.join(self.xy_dir,"X_train.npy"), arr=X_train)
            np.save(os.path.join(self.xy_dir,"Y_train.npy"), arr=Y_train)
            np.save(os.path.join(self.xy_dir,"X_val.npy"), arr=X_val)
            np.save(os.path.join(self.xy_dir,"Y_val.npy"), arr=Y_val)           
            np.save(os.path.join(self.xy_dir,"X_test.npy"),  arr=X_test)
            np.save(os.path.join(self.xy_dir,"Y_test.npy"),  arr=Y_test)
        else:
            print("Saving XY"," "*30,end='\r')
            np.save(os.path.join(self.xy_dir,"X.npy"), arr=X)
            np.save(os.path.join(self.xy_dir,"Y.npy"), arr=Y)
        print("Dataset Compiled"," "*30)
        
    def load_XY(self,is_split=False,normalize=False,spec_masking=False):
        print("Loading sets..", end= '\r')
        if is_split:
            X_train = np.load(os.path.join(self.xy_dir,"X_train.npy"))
            Y_train = np.load(os.path.join(self.xy_dir,"Y_train.npy"))
            X_val   = np.load(os.path.join(self.xy_dir,"X_val.npy"))
            Y_val   = np.load(os.path.join(self.xy_dir,"Y_val.npy"))
            X_test  = np.load(os.path.join(self.xy_dir,"X_test.npy"))
            Y_test  = np.load(os.path.join(self.xy_dir,"Y_test.npy"))
            print("X_train:" ,np.shape(X_train))
            print("Y_train:" ,np.shape(Y_train))
            print("X_val:"   ,np.shape(X_val))
            print("Y_val:"   ,np.shape(Y_val))
            print("X_test:"  ,np.shape(X_test))
            print("Y_test:"  ,np.shape(Y_test))
            self.input_shape = np.shape(X_train[0])
            self.num_classes = len(Y_test[0])
            self.class_weights = get_class_weights(Y_train)
            print("Dataset Loaded.")
            if normalize :
                X_train = map_norm(X_train,-80,0)
                X_val   = map_norm(X_val,-80,0)
                X_test  = map_norm(X_test, -80,0) 
            if spec_masking:
                X_train = spec_mask(X_train,low_bin=4,high_bin=106,null=0,flip=True)
                X_val   = spec_mask(X_val,low_bin=4,high_bin=106,null=0,flip=True)
                X_test  = spec_mask(X_test,low_bin=4,high_bin=106,null=0,flip=True)
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
        else:
            X = np.load(os.path.join(self.xy_dir,"X.npy"))
            Y = np.load(os.path.join(self.xy_dir,"Y.npy"))
            print("X:", np.shape(X))
            print("Y:", np.shape(Y))
            self.input_shape = np.shape(X[0])
            self.num_classes = len(Y[0])
            self.class_weights = get_class_weights(Y)    
            if normalize : X = map_norm(X,-80,0)
            if spec_masking : X = spec_mask(X,low_bin=4,high_bin=106,null=0,flip=True)
            print("Dataset Loaded!")
            return X, Y
        
class Classifier:
    def __init__(self, name="new_model", model=None, labels=None):
        self.name       = name
        self.model      = model
        self.labels     = labels
        self.result_dir = "results"
        self.models_dir = "models"
        if not os.path.isdir(self.result_dir): os.mkdir(self.result_dir)
        if not os.path.isdir(self.models_dir): os.mkdir(self.models_dir)
        print("Classifier Created!")  
        
    def set_name(self, name):
        self.name = name
        
    def set_labels(self, labels):
        self.labels = labels
        
    def load(self,model_path):
        self.model = keras.models.load_model(model_path)
        
    def save_summary(self):
        path = os.path.join(self.result_dir,time_stamp()+"_"+self.name+"_summary.txt")
        with open(path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary()      
       
    def save_as_keras_model(self):
        self.model.save(os.path.join(self.models_dir,time_stamp()+"_"+self.name +"_keras.hdf5"))
        
    def save_as_tflite(self,keras_model_path): # works better with nightly build
        print('Converting to TensorFlow Lite',end='\r')
        converter = lite.TFLiteConverter.from_keras_model_file(keras_model_path)
        tflite_model = converter.convert()
        open(os.path.join(self.models_dir,self.name + ".tflite"),"wb").write(tflite_model)
        print(self.name+".tflite",'created'," "*50,end='\r')
                      
    def set_callbacks(self, model_checkpoint=True, csv_log=True, tensorboard=False, reduce_plateau=True):
        print("Setting callbacks",(" "*20),end='\r')
        callbacks_list = []
        if model_checkpoint:
            checkpoint_dir = os.path.join(self.models_dir,self.name + "_weights")
            formating = "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
            filepath = os.path.join(checkpoint_dir,formating)
            if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
            checkpoint_callback = keras.callbacks.ModelCheckpoint( 
                filepath = filepath,
                monitor='val_acc',
                save_best_only=True)
            callbacks_list.append(checkpoint_callback)
        if tensorboard:
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir ='tensorboard_log/{}'.format(self.name+ "-{}".format(int(time.time()))),
                histogram_freq = 0,
                write_graph = True,
                write_images = False)
            callbacks_list.append(tensorboard_callback)
        if reduce_plateau:
            reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                verbose=0,
                mode='auto',
                min_delta=0.0001,
                cooldown=0,
                min_lr=0)
            callbacks_list.append(reduce_lr_callback)
        if csv_log:   
            csv_dir = "CSV_logs"
            if not os.path.isdir(csv_dir): os.mkdir(csv_dir)
            csv_log_callback = keras.callbacks.CSVLogger(os.path.join(csv_dir,self.name+'_training_log.csv'),append=True)
            callbacks_list.append(csv_log_callback)
        print("Calls OK",(" "*20),end='\r')
        return callbacks_list
    
    def plot_learning(self):
        df = pd.read_csv("CSV_logs\\"+self.name+'_training_log.csv',sep=',', header = 0)
        fig = plt.figure(figsize=(7,5))
        ax1 = fig.add_subplot(111)
        ax1.grid(linestyle='-.', linewidth='0.5', color='gray')
        ax2 = ax1.twinx() 
        ax1.plot(df.values[:,1],color='powderblue')
        ax2.plot(df.values[:,2],color='wheat')
        ax1.plot(df.values[:,4],color='deepskyblue')
        ax2.plot(df.values[:,5],color='orange')
        acc_max = np.max(df.values[:,4])
        acc_index = list(df.values[:,4]).index(acc_max)
        loss_min = np.min(df.values[:,5])
        loss_index = list(df.values[:,5]).index(loss_min)
        plt.title(self.name+" learning log"+" (best:"+str(loss_index+1)+")")
        ax1.set_xlabel("epochs")
        ax1.set_ylabel('accuracy',color=[0.3,0.6,0.9])
        ax2.set_ylabel('loss',color='orange')
        ax1.legend(['acc','val_acc'],loc='upper left')
        ax2.legend(['loss','val_loss'],loc='lower left')
        ax1.tick_params(axis='y', colors=[0.3,0.6,0.9])
        ax2.tick_params(axis='y', colors='orange')
        acc_text="val_acc:"+str(np.round(acc_max,2))+"↑"
        loss_text="val_loss:"+str(np.round(loss_min,2))+"↓"
        acc_box  = dict(boxstyle='round',facecolor='lightskyblue',alpha=0.5)
        loss_box = dict(boxstyle='round',facecolor='orange',alpha=0.5)
        fontdict = dict(horizontalalignment='right')
        ax1.text(acc_index,acc_max-0.05,acc_text,bbox=acc_box,fontdict=fontdict)
        ax2.text(loss_index,loss_min+0.1,loss_text,bbox=loss_box,fontdict=fontdict)
        savename=os.path.join(self.result_dir,time_stamp()+"_"+self.name+"_learning.png")
        plt.savefig(savename,dpi=300,bbox_inches='tight')
        
    def test(self, X_test, Y_test):
        if self.model == None :
            print("Load Model First")
            return
        print("Computing Predictions..",end='\r')
        output_tensors = self.model.predict(x=X_test)
        predictions = np.argmax(output_tensors,axis = 1)
        print("Computing Probabilities..",end='\r')
        true = np.argmax(Y_test,axis = 1)
        accuracy = str(round((list(predictions == true).count(True)/len(predictions))*100,2))+"%"
        print("Accuracy:",accuracy," "*40)
        conf_report = sklearn.metrics.classification_report(true,predictions,target_names=self.labels)
        save_text(conf_report,os.path.join(self.result_dir,time_stamp()+"_"+self.name+"_report"))
        self.save_summary()
        print(conf_report)
        return output_tensors, predictions, true
    
    def analysis(self,output_tensors, predictions, true):
        fig = plt.figure(figsize=(15,3))
        # confussion matrix:
        ax1 = fig.add_subplot(131)
        label_indeces = list(range(0,len(self.labels)))
        conf = sklearn.metrics.confusion_matrix(y_true=true,y_pred=predictions,labels=label_indeces)
        support = [np.sum(conf[ind,:]) for ind in label_indeces]
        conf=(np.round(conf/support,2)*100).astype(int)
        im = ax1.imshow(conf,cmap = 'Blues',vmin=0,vmax=100)
        ax1.set_xticks(np.arange(len(self.labels)))
        ax1.set_yticks(np.arange(len(self.labels)))
        ax1.set_xticklabels(self.labels)
        ax1.set_yticklabels(self.labels)
        plt.setp(ax1.get_xticklabels(), rotation=90, va="baseline", ha="center",rotation_mode="default")
        for i in range(len(self.labels)):
            for j in range(len(self.labels)):
                text = ax1.text(j, i, conf[i, j],ha="center", va="center", color="w")
        cbar = ax1.figure.colorbar(im, ax=ax1)
        ax1.set_title("Confusion Matrix")
        # roc curves:
        ax2 = fig.add_subplot(132)
        AUC = []
        labels_indeces = np.arange(len(self.labels))
        for label in labels_indeces:       
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                y_true            = true,
                y_score           = output_tensors[:,label],
                pos_label         = label, 
                drop_intermediate = False)
            AUC.append(sklearn.metrics.auc(fpr, tpr))
            ax2.plot(fpr, tpr, c=random_RGB(len(self.labels))[label],linewidth=0.5) 
        ax2.plot([0, 1],[0, 1],color='k',linewidth=1,linestyle='--')
        ax2.set_title(label ='Receiver Operator Characteristic (ROC)')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.legend(labels = self.labels, loc='lower right',prop={'size': 8})
        sns.despine()
        # area under the curve barplot:
        ax3 = fig.add_subplot(133)
        ax3.bar(labels_indeces,AUC,color=random_RGB())
        ax3.set_xlabel('Labels')
        ax3.set_ylabel('Area Under The Curve')
        ax3.set_xticklabels(self.labels,rotation='vertical',minor=True)
        sns.despine()
        fig.savefig(os.path.join(self.result_dir,time_stamp()+"_"+self.name+"_analysis"),bbox_inches='tight', dpi=300) 
        
    def audio_predictor(self,evaluation_dir,feature_type,normalize=False,spec_masking=False):
        if not os.path.isdir("Batch Evaluation Results"): os.mkdir("Batch Evaluation Results")
        save_dir = os.path.join("Batch Evaluation Results",self.name)
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        folders = os.listdir (evaluation_dir)
        cummulative_master = []
        for folder in folders:  
            folder_predictions = np.zeros([len(self.labels)])
            folder_path = os.path.join(evaluation_dir,folder)                       
            audio_files = os.listdir(folder_path)                  
            for audio_file in audio_files:   
                progress(folder,folders,subelement=audio_file)                
                y=load_audio(os.path.join(folder_path,audio_file) )
                full_spec = melspectrogram(y)
                clips = split_audio_clips(y,clip_length=132096)
                audio_predictions=np.zeros([len(self.labels)])
                predictiongram = []
                for clip_count, clip in enumerate(clips):
                    feature = feature_extract(clip,feature_type)  
                    feature = feature_reshape(feature,feature_type)
                    if normalize:
                        feature = map_norm(data=feature,data_min=-80,data_max=0)
                    if spec_masking:
                        feature = spec_mask(feature,4,106,flip=True)
                    out = self.model.predict(feature)[0]
                    audio_predictions += keras.utils.to_categorical(np.argmax(out),num_classes=len(self.labels))
                    predictiongram.append(out)
                total_samples = int(np.sum(audio_predictions))
                gs = gridspec.GridSpec(nrows= 2, ncols= 2,wspace=0.5)
                fig = plt.figure(figsize=(10,8))
                plt.subplot(gs[0,0])
                plt.plot(y,color='darkblue')
                plt.xlim([0,len(y)])
                plt.xticks([])
                plt.title(audio_file)
                plt.xlabel("Time")
                plt.ylabel("Amplitud")
                plt.subplot(gs[0,1])
                ax = plt.barh(self.labels,audio_predictions,color='darkblue',align='center')
                plt.xlim([0,np.sum(audio_predictions)])
                plt.title("Predicciones")
                plt.subplot(gs[1,0]) 
                im = plt.imshow(np.flipud(np.transpose(predictiongram)),cmap='YlGnBu',interpolation='hanning',aspect='auto')
                plt.title("Predictograma")
                plt.xlabel("Frames")
                plt.yticks(np.arange(len(self.labels)),self.labels[::-1])
                cbar = plt.colorbar(im,shrink=1,panchor='false',orientation="horizontal",spacing='uniform',pad=0.2)
                plt.subplot(gs[1,1])
                im = plt.imshow(full_spec,cmap='YlGnBu',interpolation='hanning',aspect='auto')
                plt.title("Espectrograma")
                plt.xticks([])
                plt.yticks([])
                plt.xlim(0,total_samples)
                plt.xlabel("Frames")
                plt.ylabel("Mels")
                fig_name = "BA_eval_"+audio_file+".png"
                plt.savefig(os.path.join(save_dir,fig_name),dpi=300,bbox_inches='tight')
                fig.clear()
                plt.close(fig)
                folder_predictions+=audio_predictions
            cummulative_master.append(folder_predictions)
        with open(os.path.join(save_dir,"predictions.csv"), mode='w') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(["Folder",*self.labels])
            for folder,predictions in zip(folders,cummulative_master):
                file_writer.writerow([folder,*predictions])
        print("Finished, saved in:",save_dir," "*50)

class DataGenerator(keras.utils.Sequence):
    def __init__(self,X,Y,batch_size,shuffle=True, spec_filter=False, spec_drop=False):
        self.X           = X
        self.Y           = Y
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.spec_filter = spec_filter
        self.spec_drop   = spec_drop
        self.n_samples   = len(X)
        self.on_epoch_end()
        
    def on_epoch_end(self):
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        batches_per_epoch = int(np.floor(self.n_samples / self.batch_size))
        return batches_per_epoch
            
    def __getitem__(self, index):
        lower_lim = index*self.batch_size
        upper_lim = (index+1)*self.batch_size
        if upper_lim > self.n_samples : upper_lim = lower_lim + (self.n_samples - upper_lim)
        batch_indexes = self.indexes[lower_lim:upper_lim]
        X_batch = np.array(self.X[batch_indexes])
        Y_batch = np.array(self.Y[batch_indexes])
        if self.spec_drop    : X_batch = spec_dropout(X_batch=X_batch,n_strips=2,min_strip_size=5) 
        if self.spec_filter  : X_batch = spec_filters(X_batch)
        return X_batch, Y_batch
