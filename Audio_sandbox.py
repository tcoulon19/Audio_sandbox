import librosa as lb
import matplotlib.pyplot as plt
import numpy as np


def audio_test():

    # Import music data
    x, Fs = lb.load('Audio_sandbox/download.wav', sr=None)
    t = np.arange(0,len(x))/Fs

    plt.figure(0)
    plt.plot(t,x)
    plt.show()

    # Quantize data
    quantization_levels = 64
    bins = np.arange(-1.0,1.0+2/(quantization_levels-1),2/(quantization_levels-1))
    x_quantized = np.digitize(x, bins)

    plt.figure(1)
    plt.plot(t,x_quantized)
    plt.show()

    # Convert quantized data into bit stream
    bit_stream_string = ""
    for i in range(len(x_quantized)):
        bit_stream_string += np.binary_repr(x_quantized[i], width=int(np.sqrt(quantization_levels)))
    bit_stream = []
    bit_stream.extend(bit_stream_string)
    bit_stream = list(map(int, bit_stream))
    

audio_test()