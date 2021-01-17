# -*- coding: utf-8 -*-
"""

@author: ajayv
"""

# Library imports

#standard libraries
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


from tqdm import tqdm

#audio processing specific library imports 
from python_speech_features import mfcc,logfbank
import librosa

#one hot encoding
from keras.utils import to_categorical



# standard plotting calls
def plot_signals(signals):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(1):
        for y in range(1):
            #axes.set_title(list(signals.keys())[i])
            axes.plot(list(signals.values())[i])
            axes.get_xaxis().set_visible(True)
            axes.get_yaxis().set_visible(True)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(1):
        for y in range(1):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            #axes.set_title(list(fft.keys())[i])
            axes.plot(freq, Y)
            axes.get_xaxis().set_visible(True)
            axes.get_yaxis().set_visible(True)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(1):
        for y in range(1):
            #axes.set_title(list(fbank.keys())[i])
            axes.imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes.get_xaxis().set_visible(True)
            axes.get_yaxis().set_visible(True)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(1):
        for y in range(1):
           # axes.set_title(list(mfccs.keys())[i])
            axes.imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes.get_xaxis().set_visible(True)
            axes.get_yaxis().set_visible(True)
            i += 1
            

        
# Step 1: Check the audio length distribution and bird sound presence class distribution


# 1.1 ff1010 bird data set
filepath_ff = os.path.join(os.getcwd(),'ff1010bird_metadata.csv')

df_ff = pd.read_csv(filepath_ff)
df_ff.set_index('itemid',inplace=True)


for f in df_ff.index:
    location = os.path.join('ff1010bird/'+str(f)+'.wav')
    rate,signal = wavfile.read(location)
    df_ff.at[f,'length'] = signal.shape[0]/rate
    df_ff.at[f,'rate'] = rate
    

# 1.2 And, warblrb10k bird data set

filepath_warblr = os.path.join(os.getcwd(),'warblrb10k_public_metadata.csv')

df_warblr = pd.read_csv(filepath_warblr)
df_warblr.set_index('itemid',inplace=True)

for f in df_warblr.index:
    location = os.path.join('warblrb10k/'+str(f)+'.wav')
    rate,signal = wavfile.read(location)
    df_warblr.at[f,'length'] = signal.shape[0]/rate
    df_warblr.at[f,'rate'] = rate
    
# 1.3
# confirm that ff1010 bird data is all 10 seconds and warblr is varying from 0.96 second to 48.413 seconds
# confirm that sampling rate is 44.1 kHz for all the sames across ff1010 and warblr datasets
print(df_ff.length.min(),df_ff.length.max(), df_ff.rate.min(), df_ff.rate.max())
print(df_warblr.length.min(),df_warblr.length.max(), df_warblr.rate.min(),df_warblr.rate.max())



# 1.4 bird sound presence class distribution
class_dist_ff = df_ff.groupby(['hasbird'])['hasbird'].count()

fig,ax = plt.subplots()
ax.set_title('class dist ff1010 data set')
ax.pie(class_dist_ff,labels=class_dist_ff.index, autopct = '%1.1f%%', 
       shadow = False, startangle=90)
ax.axis ('equal')
plt.show()

class_dist_war = df_warblr.groupby(['hasbird'])['hasbird'].count()

fig,ax = plt.subplots()
ax.set_title('class dist warblr data set')
ax.pie(class_dist_war,labels=class_dist_war.index, autopct = '%1.1f%%', 
       shadow = False, startangle=90)
ax.axis ('equal')
plt.show()


# 1.5 reset the index back
df_ff.reset_index(inplace=True)
df_warblr.reset_index(inplace=True)


    
## Step 2: Calculate Spectral density features
### STFT
### Triangular filter bank
### Mel Frequency Ceps tral Coefficients

# 2.1 find meaningful signal first: signal envelope

def signal_envelope(signal, rate, threshold):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# 2.2 calculate signal power spectral densities (frequency time domain)

# 2.2.1 Discrete fast fourier Transform
def fft_cal(signal,sig_sample_rate):
    signal_length = len(signal)
    sample_spacing = 1/sig_sample_rate
    #print("fft",signal_length,sig_sample_rate, sample_spacing)
    frequency = np.fft.rfftfreq(signal_length,d = sample_spacing)
    amp = abs(np.fft.rfft(signal)/signal_length) # magnitude of the DFT normalized over signal length
    return (amp,frequency)

# Triangular filter bank
def filter_bank(signal, sig_sample_rate, num_filters, window_size):
    #number of samples (num_blocks) in an FFT of window_size of x seconds
    num_blocks = 1/window_size
    num_fft = math.ceil(sig_sample_rate/num_blocks) # of windows in the signal of window size x ms 
    #print("filterbank", num_blocks, num_fft)
    bank = logfbank(signal,sig_sample_rate,winlen = window_size, 
                    winstep = window_size/2, nfilt=num_filters, nfft = num_fft).T
    return (bank)

# MFCC transforamtion
def mel_transform (signal, sig_sample_rate, num_filters, window_size):
    #number of samples (num_blocks) in an FFT of window_size of x seconds
    num_blocks = 1/window_size
    num_fft = math.ceil(sig_sample_rate/num_blocks) # of windows in the signal of window size x ms 
    #print("mel", num_blocks, num_fft) 
    mel = mfcc(signal, sig_sample_rate, 
               winlen = window_size,
               winstep = window_size/2,
               numcep = num_filters,
               nfilt=num_filters, nfft = num_fft).T
    return (mel)
    
# define dictionaries to store MFCC  transformation coeffs

signals ={}
FFTs = {}
fbanks = {}
mfccs = {}


location = os.path.join('ff1010bird/'+str(df_ff['itemid'][4000])+'.wav')
rate,signal = wavfile.read(location)

# Envelope the signal

mask = signal_envelope(signal, rate, 0.0005)

print("len signal pre-masked", len(signal))
signal = signal[mask]
print("len signal masked", len(signal))

# find the power spectral features across frequency temporal domain
fft = fft_cal(signal,rate)
bank = filter_bank(signal, rate, 40, 0.040) # number of filters = 40, window size = 40 ms/ 0.040 seconds
mel = mel_transform(signal, rate, 40, 0.040) # number of filters = 40, window size = 40 ms/ 0.040 seconds


signals[0] = signal
FFTs[0] = fft
fbanks[0] = bank
mfccs[0] = mel

plot_signals(signals)
plt.show()

plot_fft(FFTs)
plt.show()

plot_fbank(fbanks)
plt.show()

plot_mfccs(mfccs)
plt.show()


## Step 3: clean and resample (downsample) and store the data

# 3.1 database ff1010 bird sampled for 20kHz sampling rate and removing noise by signal envelope

if len(os.listdir('ff1010bird-cleaned')) == 0:
    for t in tqdm(df_ff.itemid):
        signal, rate = librosa.load(os.path.join('ff1010bird/'+str(t)+'.wav'),sr = 20000)
        mask = signal_envelope(signal, rate, 0.0005)
        wavfile.write(filename=os.path.join('ff1010bird-cleaned/'+str(t)+'.wav'), 
                      rate = rate, data=signal[mask])
        
# 3.2 database ff1010 bird sampled for 20kHz sampling rate and removing noise by signal envelope


# if len(os.listdir('warblrb10k-cleaned')) == 0:
#     for t in tqdm(df_warblr.itemid):
#         signal, rate = librosa.load(os.path.join('warblrb10k/'+str(t)+'.wav'),sr = 20000)
#         mask = signal_envelope(signal, rate, 0.0005)
#         wavfile.write(filename=os.path.join('warblrb10k-cleaned/'+str(t)+'.wav'), 
#                       rate = rate, data=signal[mask])


## 4. Extract MFCC features
# 4.1 First, configure some constants for later tinkering
# nfft @ 800 is derived as number of samples in a 40 msecond (1/.04 slots/second) 
    #slot  of 20kHz (20 ksamples/second) sampled data. Hence 800 = 20000/(1/0.04)
# class Config:
#     def __init__(self, num_filters = 40, nfft = 800, rate = 20000, window_size=0.040):
#         self.nfilt = num_filters
#         self.nfft = nfft
#         self.rate = rate
#         self.window_size = window_size

rate = 20000
window_size=0.040
num_filters = 40
nfft = 800
_min, _max = float ('inf'), -float('inf')        

## 4.2 Extract MFCC features at ff1010 bird sound database



X =[] # to store MFCC coeffs
y = [] # to store BAD (bird audio detection) 0/1 

i = 0
for i in tqdm(range(len(df_ff))):
    file = int(df_ff.iloc[i]['itemid'])
    rate, signal = wavfile.read(os.path.join('ff1010bird-cleaned/'+ str(file)+'.wav'))

    if (signal.shape[0] != 20*10**3*10):
        print(i, signal.shape[0])
        print("================ sample file ill sampled -- NOT 20kHz*10(seconds)-----------")
        continue
    
    x_sample = mel_transform(signal, rate, num_filters, window_size)
    _min = min(np.amin(x_sample), _min)
    _max = max(np.amax(x_sample), _max)
    
    X.append(x_sample)
    y.append(int(df_ff.iloc[i]['hasbird']))
    
    
X, y = np.array(X), np.array(y)
X = (X-_min)/(_max-_min) # normalize for the entire ND Array 7513*40*499
X = X.reshape(X.shape[0], X.shape[1],X.shape[2], 1) #for Convolutional neural network setup
#X = X.reshape(X.shape[0], X.shape[1],X.shape[2]) # for recurrent neural network

y = to_categorical(y, num_classes = 2)

# Print shapes for Convolution network
# X shape is 7513 audio files, with 40 features (mfcc) and 499 
print (X.shape, y.shape)




