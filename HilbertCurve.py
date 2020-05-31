#!/usr/bin/env python
# coding: utf-8

# In[53]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
import numpy as np
import math
from skimage import data, io
import simpleaudio as sa
from scipy.io.wavfile import read, write
from IPython.display import Audio
from PIL import Image
import matplotlib.pyplot as plt
import wavio
from scipy.fftpack import fft , fft2 ,fftshift , ifftshift , ifft2
import scipy.fftpack as fftp
from skimage.filters import gaussian
import scipy as sc
import scipy.linalg as alg


# In[54]:


#==========================================decalring finding the last 2-bit function=========================================
def last2bits(x):
    return (x & 3)


# In[55]:


#==========================================decaring finding the hiblert coordinates function - 1st implementation============
def hindex2xy(hindex,N):
    positions = np.array([[0, 0],[0, 1],[1, 1],[1, 0]])
    tmp = positions[last2bits(hindex)]
    hindex =np.right_shift(hindex,2)
    x = tmp[0]
    y = tmp[1]
    for i in range(4,N+1):
        n2 = i/2
        i =i*2
        if((last2bits(hindex))==0):
            tmp = x 
            x = y
            y = tmp
        elif((last2bits(hindex))==1):
            x = x
            y = y + n2
        elif((last2bits(hindex))==2):
            x = x + n2
            y = y + n2
        elif((last2bits(hindex))==3):
            tmp = y
            y = (n2-1) - x
            x = (n2-1) - tmp
            x = x + n2
        hindex =np.right_shift(hindex,2)
        return [x, y]


# In[56]:


#=============================================decalring finding the hiblert of nth order function==========================
def hilbertCurve(n):
    nopix = 4**n
    N = 2**n
    hilbertList = []
    for i in range(nopix):
        hilbertList.append(hindex2xy(i,N))
    return np.floor(hilbertList).astype(int)


# In[57]:


#==========================================decaring finding the hiblert coordinates function - 2nd implementation============
dummyM = np.array([[0, 0], [0, 1], [1,1], [1,0]])
def firstQuadrant(Positions):
    A = np.array([[0, 1], [1, 0]])
    return Positions.dot(A.T)

def secondQuadrant(Positions, N): # Where N is the order of the Hilbert curve that we are constructing with
    B = np.repeat(np.array([[0, 2**N]]), repeats=4**N, axis=0)
    return Positions + B

def thirdQuadrant(Positions, N):
    C = np.repeat(np.array([[2**N, 2**N]]), repeats=4**N, axis=0)
    return Positions + C

def forthQuadrant(Positions, N):
    D1 = np.repeat(np.array([[-(2**N)+1, -(2**N)+1]]), repeats=4**N, axis=0)
    D2 = np.array([[0, -1], [-1, 0]])
    D3 = np.repeat(np.array([[2**N, 0]]), repeats=4**N, axis=0)
    return ((Positions + D1).dot(D2.T)) + D3

def hilbertCurve2(N):
    firstOrder = np.array([[0, 0], [0, 1], [1,1], [1,0]])
    if N == 1:
        return firstOrder
    else:
        return np.concatenate((firstQuadrant(hilbertCurve2(N-1)), secondQuadrant(hilbertCurve2(N-1), N-1), thirdQuadrant(hilbertCurve2(N-1), N-1), forthQuadrant(hilbertCurve2(N-1), N-1)))


# In[58]:


#=======================================declaring generating frequencies based on n.of image's levels - inspired from piano===
def intensityToFrequency(level):
    freq = [55]
    for i in range(0,level):
        newVal = freq[i] * 2.0**(16.0/(level*3.0))
        freq.append(newVal)
    return freq


# In[59]:


#=====================================declaring finding order and resize image to nearest hilbert order - make square image====
def squareToHilbert(img):
    row,col = img.shape[0],img.shape[1]
    arr=[]
    newDimension=0
    for i in range(15):
        #print(i)
        val = 2**i
        arr.append(val)
        if(row<=arr[i]):
            newDimension = arr[i]
            break
    order = math.floor(math.log(newDimension,2))
    newImg = np.array(Image.fromarray(img).resize((newDimension, newDimension), Image.ANTIALIAS))
    return order,newImg


# In[60]:


#===================================declaring mapping intensities of an image to frequencies==================================
def hilbertToIntensities(img, N):
    frequencies = intensityToFrequency(256)
    audioIntensities = np.zeros(4**N)
    positions = hilbertCurve(N)
    for i in range (len(positions)):
        audioIntensities[i] = frequencies[img[positions[i][0]][positions[i][1]]]
    return audioIntensities


# In[61]:


#=================================declaring making sound from generated frequencies function==================================
def imageToSound (img, timeSec):
    N, newImg = squareToHilbert(img)
    print(N)
    freq = hilbertToIntensities(newImg, N)
    fs = 44100
    m = timeSec/0.1
    print(len(freq))
    n = 4**N
    delta = ((m - 1.0)/(n - 1.0)) * 0.1
    t = np.linspace(0, timeSec, timeSec * fs, False)
    note = 0 * t
    for i in range(len(freq)):
        t1 = np.linspace(i* delta, i * delta + 0.1, timeSec * fs, False)
        ithNote = np.sin(freq[i] * t1 * 2 * np.pi)
        note = note + ithNote
        if i % 1000 == 0:
            print(str(i/len(freq)*100) + '%')
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    audio = audio.astype(np.int16)
    return audio, note


# In[62]:


#=========================================declaring playing the output audio=================================================
def playAudio(audio):
    play_audio = sa.play_buffer(audio, 1, 2, 44100)
    play_audio.wait_done()


# In[63]:


#==========================================declaring plotting waveform of output audio=======================================
def plotAudio(audio):
    Fs, data = read(audio)
    data = data[:,0]
    Audio(data, rate=Fs)
    plt.figure()
    plt.plot(data)
    plt.xlabel('Sample Time')
    plt.ylabel('Amplitude')
    plt.title('Waveform of output Audio')
    plt.show()    


# In[64]:


# ====================================Testing Cases==============================================================
# ===============================================================================================================


# In[65]:


space = io.imread('space_noise_img.png')
plt.imshow(space,'gray',vmin=0, vmax=255 )
plt.show()


# In[66]:


man = io.imread('man_noise_img.jpg')
plt.imshow(man,'gray',vmin=0, vmax=255 )
plt.show()


# In[67]:


#========================declaring displaying in frequency domian function================
def displayFourier(img):
    freqImage=fftshift(fft2(img))
    freqImage=np.abs(freqImage)
    freqImage=20*np.log(freqImage)
    return freqImage


# In[68]:


#========================declaring inverting fourier function==============================
def invertFourier(img):
    imageBack = np.real(ifft2(img))+np.imag(ifft2(img))
    return imageBack


# In[69]:


#========================declaring plotting function=======================================
def plotComparison(original, operation,left_title, right_title):
    fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(16,8))
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title(left_title)
    ax2.imshow(operation, cmap=plt.cm.gray)
    ax2.set_title(right_title)


# In[70]:


plotComparison(displayFourier(space),invertFourier(fft2(space)),'Original','Invert Fourier')


# In[71]:


#========================declaring desiging notch filter function========================
def designNotchFilter(img):
    row = img.shape[0]
    col = img.shape[1]
    notchFilter = np.ones((row,col))
    #assign 0 to noise peaks which varies with images
    notchFilter[175:205,210:240]=0
    notchFilter[175:205,385:420]=0
    notchFilter[290:320,210:245]=0
    notchFilter[290:320,390:420]=0
    return notchFilter


# In[72]:


#========================declaring removing periodic noise function========================
def removeNoise(img):
    # 1. shifting image to frequency domain
    imgShift = fftshift(fft2(img))
    # 2. designing notch filte
    notch = designNotchFilter(img)
    # 3. multiplying notch filter by image
    mult = imgShift * notch
    # 4. returning denoised image back to spatial domain
    imageBack = ifftshift(mult)
    imageBack = np.abs(ifft2(imageBack))
    return imageBack


# In[73]:


#========================declaring displaying only noise function=========================
def displayNoise(original_img,denoised_image):
    noise = original_img - denoised_image
    return noise


# In[74]:


plotComparison(space,removeNoise(space),'Original','Denoised Space Image After Notch Filter')
plotComparison(space,displayNoise(space,removeNoise(space)),'Original','Noise Only')


# In[75]:


#========================Images for testing================================================================
space = io.imread('space_noise_img.png')
denoised_space = removeNoise(space)
checkboard = data.checkerboard()
camerman = data.camera()
grass = data.grass()
gravel = data.gravel()
clock = data.clock()
coins = data.coins()


# In[76]:


#========================resize function================================================================
def resizeToAnyDimension(img,row,col):
    newImg = np.array(Image.fromarray(img).resize((row, col), Image.ANTIALIAS))
    return newImg


# In[77]:


coins_64 = resizeToAnyDimension(coins,64,64)
camerman_64 = resizeToAnyDimension(camerman,64,64)
space_64 = resizeToAnyDimension(space,64,64)
denoised_space_64 = resizeToAnyDimension(denoised_space,64,64)
grass_64 = resizeToAnyDimension(grass,64,64)
gravel_64 = resizeToAnyDimension(gravel,64,64)
clock_64 = resizeToAnyDimension(clock,64,64)


# In[52]:


#=================================Cases for testing======================================================================


# In[ ]:


# outSound, x = imageToSound(coins_64, 60)
# wavio.write('output1.wav', outSound, 44100)


# In[ ]:


# outSound, x = imageToSound(camerman_64, 60)
# wavio.write('output2.wav', outSound, 44100)


# In[ ]:


# outSound, x = imageToSound(space_64, 60)
# wavio.write('output3.wav', outSound, 44100)


# In[ ]:


# outSound, x = imageToSound(denoised_space_64, 60)
# wavio.write('output4.wav', outSound, 44100)


# In[ ]:


# outSound, x = imageToSound(grass_64, 60)
# wavio.write('output5.wav', outSound, 44100)


# In[31]:


# outSound, x = imageToSound(gravel_64, 60)
# wavio.write('output6.wav', outSound, 44100)


# In[32]:


# outSound, x = imageToSound(clock_64, 60)
# wavio.write('output7.wav', outSound, 44100)


# In[ ]:


# t = np.linspace(0, 60, 60* 44100, False)
# plt.plot(t, x) 
# plt.show()

