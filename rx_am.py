# opencv2-python
# numpy

import cv2
import numpy as np
import spidev

############################################################################
# Pre-define the Parameters #

np.random.seed(42)
img = cv2.imread('/Lena_color.png')
resized_img = cv2.resize(img, (128, 128),cv2.INTER_CUBIC)
img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
threshold, binarized_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
binarized_img = binarized_img/255

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

# Quantization
n_bits = 5  
quantization_levels = 2 ** n_bits

############################################################################

time = 100000 # 90000*0.016 receiving
spi = spidev. SpiDev()
spi.open(0,0)
spi.max_speed_hz = 100000

def ReadChannel (channel):
  adc = spi.xfer([6| (channel & 4) >> 2, (channel & 3) << 6 , 0])
  data = ((adc[1] & 15) << 8) + adc[2]
  return data

rx_signal = []
for i in range(time):
  rx_signal.append(ReadChannel(0))

rx_signal = np.array(rx_signal[4000:]) # Remove SPI Bottleneck
rx_signal -= np.mean(rx_signal) # Remove DC Offset

# Pulse Judgement
# If pulse amplitude is bigger than half value of maximum signal, it becomes 1; else 0
for i, signal in enumerate(rx_signal):
  if signal >= np.max(rx_signal)/2:
    rx_signal[i] = 1
  else:
    rx_signal[i] = 0

def pcm_decode(binary_data, n_bits, quantization_levels):
    """PCM 디코딩 함수"""
    decoded_values = [int(b, 2) for b in binary_data]  # 이진수 -> 정수 변환
    decoded_signal = 2 * (np.array(decoded_values) / (quantization_levels - 1)) - 1  # 원래 값 복원
    return decoded_signal

#decoded_values = np.array([int(binary.strip(), 2) for binary in tx_signal])
rx_signal = pcm_decode(rx_signal, n_bits, quantization_levels)

xC = np.correlate(rx_signal,preamble, mode='full')
lags = np.arange(-len(rx_signal), len(rx_signal))
start_pt= lags[np.argmax(xC)]
rx_signal = rx_signal[start_pt+1:]

for i, signal in enumerate(rx_signal):
  if signal >= 0.5:
    rx_signal[i] = 1
  else:
    rx_signal[i] = 0

decoded_bits = rx_signal.reshape(int(np.sqrt(len(rx_signal))) ,int(np.sqrt(len(rx_signal))))
ber = np.sum(decoded_bits.reshape(-1,1) != np.array(binarized_img).reshape(-1,1)) / 16384
print(ber)