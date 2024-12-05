# opencv2-python
# numpy
# sounddevice
# scipy
import board
import busio
import adafruit_mcp4725
import cv2
import numpy as np

np.random.seed(42) # Seed Fix
img = cv2.imread('/Lena_color.png')
resized_img = cv2.resize(img, (128, 128),cv2.INTER_CUBIC)
img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
threshold, binarized_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
binarized_img = binarized_img/255

# STEP 1
bits = list(binarized_img.reshape(-1))

# interleaving

elements = np.random.permutation(len(bits))
bits = bits[elements]

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
half_N = N / 2
# 양의 주파수 부분을 무작위로 1과 -1로 채우기 (Improved Pilot signal)
positive_freq_part = (np.random.rand(half_N , 1) > 0.5) * 2 - 1
# Hermitian 대칭 벡터 생성 (첫 번째와 중간 값 제외)
pilot_freq = [0, positive_freq_part,  np.flip(np.conj(positive_freq_part[:N/2-1]))]
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



# Initialize I2C bus.
i2c = busio.I2C(board.SCL, board.SDA)
dac = adafruit_mcp4725.MCP4725(i2c)

# Tx
# Send trash value while SPI Setup
for i in range(4000):
   dac.normalized_value = 0.1
  
for t in tx_signal:
   dac.noramlized_value = (t+10)/20 # Min-Max Scaler # Max: 10 / # Min: -10
