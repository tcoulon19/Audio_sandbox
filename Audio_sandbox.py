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
    plt.title('Audio signal, sampled')
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

    # Channel and demodulation
    for i,EbN0 in enumerate(EbN0dB):
            
            # Compute and add AWGN noise
            r = awgn(s, EbN0, L) # Refer Chapter section 4.1

            r_bb = r*np.cos(2*np.pi*Fc*t/Fs) # Recovered baseband signal
            ak_hat = bpsk_demod(r_bb, L) # Baseband correlation demodulator
            BER[i] = np.sum(bit_stream != ak_hat)/len(bit_stream) # Bit Error Rate Computation (!= means "not equal to")

            # Received signal waveform zoomed to first 10 bits, EbN0dB=9
            if EbN0 == 10:

                plt.figure(4)
                plt.clf()
                plt.plot(t,r)
                plt.xlabel('t(s)')
                plt.ylabel('r(t)')
                plt.xlim(0,10*L)
                plt.title('Recieved signal with noise, EbN0=10')
                plt.show()

                plt.figure(5)
                plt.clf()
                plt.plot(16*np.arange(len(bit_stream)),ak_hat)
                plt.xlabel('t(s)')
                plt.ylabel('ak_hat')
                plt.xlim(0,10*L)
                plt.title('Demodulated signal, EbN0=10')
                plt.show()

                plt.figure(6)
                plt.plot(np.real(r), np.imag(r), 'o')
                plt.xlim(-1.5,1.5)
                plt.ylim(-1.5,1.5)
                plt.title('Constellation diagram')
                plt.show()

        #----------Theoretical Bit/Symbol Error Rates----------
    theoreticalBER = 0.5*erfc(np.sqrt(10**(EbN0dB/10))) # Theoretical bit error rate
        
        #----------Plots----------
    plt.figure(7)
    plt.clf()
    plt.semilogy(EbN0dB, BER, 'k*', label='Simulated') # Simulated BER
    plt.semilogy(EbN0dB, theoreticalBER, 'r-', label='Theoretical')
    plt.xlabel('$E_b/N_0$ (dB)')
    plt.ylabel('Probability of Bit Error - $P_b$')
    plt.title('Probability of Bit Error for BPSK modulation')
    plt.show()



audio_test()