import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
from passband_modulations import bpsk_mod, bpsk_demod
from channels import awgn
from scipy.special import erfc


def audio_test():

    # Import music data
    x, Fs = lb.load('Audio_sandbox/download.wav', sr=None)
    t = np.arange(0,len(x))/Fs

    plt.figure(0)
    plt.plot(t,x)
    plt.title('audio signal, sampled')
    plt.show()

    # Quantize data
    quantization_levels = 64
    bins = np.arange(-1.0,1.0+2/(quantization_levels-1),2/(quantization_levels-1))
    x_quantized = np.digitize(x, bins)

    plt.figure(1)
    plt.plot(t,x_quantized)
    plt.title('Audio signal, quantized')
    plt.show()

    # Convert quantized data into bit stream
    bit_stream_string = ""
    for i in range(len(x_quantized)):
        bit_stream_string += np.binary_repr(x_quantized[i], width=int(np.sqrt(quantization_levels)))
    bit_stream = []
    bit_stream.extend(bit_stream_string)
    bit_stream = list(map(int, bit_stream))
    bit_stream = np.asarray(bit_stream)

    # Modulation
    EbN0dB = np.arange(-4,11,2) # Eb/N0 range in dB for simulation
    L=16 # Oversampling factor, L = Tb/Ts (Tb = bit period, Ts = sampling period)
    # If carrier is used, use L = Fs/Fc, where Fs >> 2xFc
    Fc = 800 # Carrier frequency
    Fs = L*Fc # Sampling frequency
    BER = np.zeros(len(EbN0dB)) # For BER values for each Eb/N0
    (s_bb, t) = bpsk_mod(bit_stream,L) # BPSK modulation (waveform) - baseband
    s = s_bb*np.cos(2*np.pi*Fc*t/Fs) # With carrier

    plt.figure(2)
    plt.plot(t, s_bb) # Baseband wfm zoomed to first 10 bits
    plt.xlabel('t(s)')
    plt.ylabel('$s_{bb}(t)$-baseband')
    plt.xlim(0,10*L)
    plt.title('Signal after BPSK')
    plt.show()

    plt.figure(3)
    plt.plot(t, s) # Transmitted wfm zoomed to first 10 bits
    plt.xlabel('t(s)')
    plt.ylabel('s(t)-with carrier')
    plt.xlim(0,10*L)
    plt.title('Signal multiplied by carrier')
    plt.show()

    plt.figure(4)
    plt.plot(np.real(s_bb), np.imag(s_bb), 'o')
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.title('Constellation diagram')
    plt.show()


audio_test()