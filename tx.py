# opencv2-python
# numpy
# sounddevice
# scipy

import sounddevice as sd
import cv2
import numpy as np

img = cv2.imread('Please enter the address of image')
resized_img = cv2.resize(img, (128, 128),cv2.INTER_CUBIC)
img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
threshold, binarized_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
binarized_img = binarized_img/255

# STEP 1
bits = list(binarized_img.reshape(-1))
symbols =[]
for i in bits:
  symbols.append(i*2-1)

# Parameter 
M = len(symbols)
N = 256 #number of subcarriers
N_cp = 32; # number of cp
cn = int(M/(N/2))
N_blk = cn+cn//4

symbols_freq = []
for i  in range(1,cn+1):
  symbols_freq.append([0])
  symbols_freq[-1].extend(symbols[N//2*(i-1):N//2*i])
  symbols_freq[-1].extend(list(np.flip(np.array(symbols[N//2*(i-1):N//2*i-1]))))

# FFT
symbols_time = []
for i in range(len(symbols_freq)):
  symbols_time.append(np.fft.ifft(symbols_freq[i], N)*np.sqrt(N))

# Insert Cyclic Prefix
for i in range(len(symbols_time)):
  cp = symbols_time[i][-N_cp:]
  tmp =  list(symbols_time[i])
  symbols_time[i] = list(cp)
  symbols_time[i].extend(tmp)

# Pilot Signal
pilot_freq = np.ones(N)
pilot_time = list(np.fft.ifft(pilot_freq))
pilot_cp = pilot_time[-N_cp:]
pilot_cp.extend(pilot_time)
pilot_time = pilot_cp

# Preamble
omega = 10
mu = 0.1
Tp = 1000
tp = np.array([i for i in range(1, Tp+1)])
preamble = np.cos(omega*tp + 0.5*mu*(tp**2))


# Serial
tx_signal = preamble
for i in range(len(symbols_time)):
  if i % 4 == 0:
    tx_signal = np.concatenate((tx_signal, pilot_time))
  tx_signal = np.concatenate([tx_signal, symbols_time[i]])
  tx_signal = np.real(tx_signal)


fs = 10000
sd.play(tx_signal, fs)