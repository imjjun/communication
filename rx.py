# opencv2-python
# numpy
# sounddevice
# scipy

import sounddevice as sd
import cv2
import numpy as np
from scipy.signal import correlate

############################################################################
# Pre-define the Parameters #

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

# Preamble
omega = 10
mu = 0.1
Tp = 1000
tp = np.array([i for i in range(1, Tp+1)])
preamble = np.cos(omega*tp + 0.5*mu*(tp**2))

############################################################################

samplerate = 10000  # sampling rate
channels = 1  # mono = 1, stereo = 2
blocksize = 1024  # framesperblock
duration = 5 # (sec)

print("Start Recording")
audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='int8')
sd.wait()  # stop recording
print("Recording complete.")
rx_signal = ((audio_data[:, None] & (1 << np.arange(8)[::-1])) > 0).astype(int) # int8 -> binary


xC = correlate(rx_signal,np.concatenate([preamble,np.zeros(len(rx_signal)-len(preamble))]))
lags = np.arange(-len(rx_signal)+1, len(rx_signal))
start_pt= lags[np.argmax(xC)]
rx_signal = rx_signal[start_pt+Tp:]

# Delete CP
OFDM_blks = []
for i in range(N_blk):
  OFDM_blks.append(rx_signal[N_cp:N+N_cp])
  rx_signal = rx_signal[N_cp+N:]


# FFT
  demod_OFDM_blks = []
for i in range(len(OFDM_blks)):
  demod_OFDM_blks.append(np.fft.fft(OFDM_blks[i],N)/np.sqrt(N))
demod_OFDM_blks = np.array(demod_OFDM_blks)


# Channel Equalization
symbols_eq = []
for i in range(len(demod_OFDM_blks)):
  if i % 5 == 0:
    channel = demod_OFDM_blks[i]/np.ones(N)
  else:
    symbols_eq.append(demod_OFDM_blks[i]/channel)

# Symbol Detection
symbols_detect = []
for i in range(len(symbols_eq)):
  symbols_detect.append(list(np.sign(np.real(symbols_eq[i]))))

# BPSK
symbols_est = []
for i in range(len(symbols_detect)):
  symbols_est.extend(symbols_detect[i][1:N//2+1])


decoded_bits = (np.array( (symbols_est))+1)/2
decoded_bits = decoded_bits.reshape(int(np.sqrt(len(decoded_bits))) ,int(np.sqrt(len(decoded_bits))))
ber = np.sum(decoded_bits.reshape(-1,1) != np.array(bits).reshape(-1,1)) / 16384
print(ber)