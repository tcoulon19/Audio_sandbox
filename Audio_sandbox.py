import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
from passband_modulations import bpsk_mod, bpsk_demod, qpsk_mod, qpsk_demod
from channels import awgn
from scipy.special import erfc
import soundfile as sf


def audio_test():

    # Import music data
    x, Fs = lb.load('Audio_sandbox/Recording.wav', sr=None)
    t = np.arange(0,len(x))/Fs

    plt.figure(0)
    plt.plot(t,x)
    plt.title('Audio signal, sampled')
    plt.ylim(-1,1)
    plt.show()

    # Quantize data
    quantization_levels = 32
    bins = np.arange(-1.0,1.0+2/(quantization_levels-1),2/(quantization_levels-1))
    x_quantized = np.digitize(x, bins)

    plt.figure(1)
    plt.plot(t,x_quantized)
    plt.title('Audio signal, quantized')
    plt.ylim(0,quantization_levels)
    plt.show()

    # Convert quantized data into bit stream
    bit_stream_string = ""
    for i in range(len(x_quantized)):
        bit_stream_string += np.binary_repr(x_quantized[i], width=int(np.sqrt(quantization_levels)))
    bit_stream = []
    bit_stream.extend(bit_stream_string)
    bit_stream = list(map(int, bit_stream))
    bit_stream = np.asarray(bit_stream)

    # Modulation (QPSK)
    EbN0dB = np.arange(-4,11,2) # Eb/N0 range in dB for simulation
    fc = 100 # Carrier frequency in Hz
    OF = 8 # Oversampling factor, sampling frequency will be fs = OF*fc
    BER = np.zeros(len(EbN0dB)) # For BER values for each Eb/N0
    result = qpsk_mod(bit_stream,fc,OF,enable_plot=False,enable_plot_const=False) # QPSK modulation
    s = result['s(t)'] # Get values from returned dictionary

    for i, EbN0 in enumerate(EbN0dB):

        # Compute and add AWGN noise
        r = awgn(s,EbN0,OF) # Refer Chapter section 4.1
        
        if EbN0 == 8:
            a_hat = qpsk_demod(r,fc,OF,enable_plot=False) # QPSK demodulation
            bit_stream_rx = np.asarray(a_hat)
        else:
            a_hat = qpsk_demod(r,fc,OF,enable_plot=False)

        BER[i] = np.sum(bit_stream != a_hat)/len(bit_stream) # Bit error rate computation

    #--------Theoretical bit error rate--------
    theoreticalBER = .5*erfc(np.sqrt(10**(EbN0dB/10)))

    #--------Plot performance curve--------
    plt.figure(9)
    plt.clf()
    plt.semilogy(EbN0dB, BER, 'k*', label='Simulated')
    plt.semilogy(EbN0dB, theoreticalBER, 'r-', label='Theoretical')
    plt.xlabel('$E_b/N_0$ (dB)')
    plt.ylabel('Probability of Bit Error - $P_b$')
    plt.title('Probability of Bit Error for QPSK modulation')
    plt.legend()
    plt.show()

    rx_quantized = []
    for i in range(len(x_quantized)):
        bit_stream_rx_symb = \
            bit_stream_rx[i*int(np.sqrt(quantization_levels)):i*int(np.sqrt(quantization_levels))+\
            int(np.sqrt(quantization_levels))]
        bit_stream_rx_symb = list(map(int, bit_stream_rx_symb))
        bit_stream_rx_symb = list(map(str, bit_stream_rx_symb))
        bit_stream_rx_symb = "".join(bit_stream_rx_symb)
        rx_quantized.append(int(bit_stream_rx_symb,2))

    plt.figure(10)
    plt.plot(t,rx_quantized)
    plt.title('Audio signal received, quantized')
    plt.ylim(0,quantization_levels)
    plt.show()

    rx = [(val - quantization_levels/2)/(quantization_levels/2) for val in rx_quantized]

    plt.figure(11)
    plt.plot(t,rx)
    plt.title('Audio signal received')
    plt.ylim(-1,1)
    plt.show()

    sf.write('Audio_sandbox/Received_audio.wav', rx, Fs)


audio_test()