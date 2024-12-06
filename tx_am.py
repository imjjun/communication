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

# Quantization
n_bits = 5  # 3-bits quantizaiton: Similiar to Repetition code 
quantization_levels = 2 ** n_bits
normalized_image = (binarized_img / 255) * (quantization_levels - 1)  # 0~255 -> 0~255
quantized_image = np.round(normalized_image).astype(int)  # integer
binary_encoded = [
    format(pixel, f'0{n_bits}b') for pixel in quantized_image.flatten()
]

# Preamble
omega = 10
mu = 0.1
Tp = 1000
tp = np.array([i for i in range(1, Tp+1)])
preamble = np.cos(omega*tp + 0.5*mu*(tp**2))

def pcm_encode(signal, n_bits):
    # Make it not negative
    quantized_signal = np.round(((signal +1) / 2) * (quantization_levels - 1))
    return quantized_signal.astype(int)

quantized_preamble = pcm_encode(preamble, n_bits)
binary_preamble = [
    format(val, f'0{n_bits}b') for val in quantized_preamble.flatten()
]

tx_signal = binary_preamble + binary_encoded

# Initialize I2C bus.
i2c = busio.I2C(board.SCL, board.SDA)
dac = adafruit_mcp4725.MCP4725(i2c, address = '0x62')

# Tx
# Send trash value while SPI Setup
for i in range(4000):
   dac.raw_value = 1

# Data
# 5 * 17384
for t in tx_signal:
    for i in t:
        if i == '1':
            dac.raw_value = 4095
        else:
            dac.raw_value = 1
   
