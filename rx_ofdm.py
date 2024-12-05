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

# Pilot
half_N = N // 2
positive_freq_part = (np.random.rand(half_N - 1, 1) > 0.5) * 2 - 1  # 첫 번째와 중간 값 제외
# Hermitian 대칭 벡터 생성
pilot_freq = np.ones((N,))  # N 크기의 복소수 배열 생성
pilot_freq[1:half_N] = positive_freq_part.flatten()  # 양의 주파수 부분 채우기
pilot_freq[-(half_N - 1):] = np.conj(positive_freq_part[::-1].flatten())  # Hermitian 대칭

############################################################################

time = 60000 # 60000*0.016 동안 수신
spi = spidev. SpiDev()
spi.open(0,0)
spi.max_speed_hz = 100000

def ReadChannel (channel):
  adc = spi.xfer([6| (channel & 4) >> 2, (channel & 3) << 6 , 0])
  data = ((adc[1] & 15) << 8) + adc[2]
  return data

rx_signal = []
for i in range(time):
  rx_signal.append((ReadChannel(0)+10)/20)

rx_signal = np.array(rx_signal[4000:]) # Remove SPI Bottleneck
rx_signal -= np.mean(rx_signal) # Remove DC Offset

xC = np.correlate(rx_signal,preamble, mode='full')
lags = np.arange(-len(rx_signal), len(rx_signal))
start_pt= lags[np.argmax(xC)]
rx_signal = rx_signal[start_pt+1:start_pt+1+46080]

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
    channel = demod_OFDM_blks[i]/pilot_freq
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

elements = np.random.permutation(len(bits))
deinterleaved_bits = np.zeros_like(bits)
deinterleaved_bits[elements] = decoded_bits
decoded_bits = deinterleaved_bits

decoded_bits = decoded_bits.reshape(int(np.sqrt(len(decoded_bits))) ,int(np.sqrt(len(decoded_bits))))
ber = np.sum(decoded_bits.reshape(-1,1) != np.array(binarized_img).reshape(-1,1)) / 16384
print(ber)