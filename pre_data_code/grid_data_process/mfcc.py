import os
import shutil
import numpy as np
import math

import python_speech_features
from python_speech_features import sigproc
from scipy.io import wavfile
from scipy.fftpack import dct


def ceplifter_coefficients(N, L):
    n = np.arange(N)
    lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
    return lift


# Slightly modified function of the python_speech_features package
# https://github.com/jameslyons/python_speech_features/blob/40c590269b57c64a8c1f1ddaaff2162008d1850c/python_speech_features/base.py#L149
def get_filterbanks_alt(nfilt=20, nfft=1024, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = python_speech_features.hz2mel(lowfreq)
    highmel = python_speech_features.hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt+2)
    f = np.linspace(0, 0.5 * samplerate, nfft//2+1)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    # bin = np.floor((nfft+1)*python_speech_features.mel2hz(melpoints)/samplerate)
    bin = python_speech_features.mel2hz(melpoints)

    fbank = np.zeros([nfilt, nfft//2+1])
    for j in range(0, nfilt):
        # up-slope
        k = np.where((f >= bin[j]) & (f <= bin[j+1]))
        fbank[j, k] = (f[k] - bin[j])/(bin[j+1]-bin[j])
        # down-slope
        k = np.where((f >= bin[j+1]) & (f <= bin[j+2]))
        fbank[j, k] = (bin[j+2] - f[k]) / (bin[j+2] - bin[j+1])

    return fbank


def wav_to_mfcc(wav_file, save_mfcc_dir):

    sample_rate, audio = wavfile.read(wav_file)
    # to get same numbers as returned by respective matlab script
    # sample_rate, audio0 = wavfile.read(audiotmp)
    # audio = audio0 / 32767.
    # scale is not important though

    try:
        audio = audio[:, 0]
    except IndexError:
        pass

    signal = audio
    samplerate = sample_rate
    winlen = 0.025
    winstep = 0.01
    numcep = 13
    nfilt = 13
    nfft = 512
    lowfreq = 300
    highfreq = 3700
    preemph = 0.97
    ceplifter = 22
    # appendEnergy = True
    winfunc = np.hamming

    # python_speech_features.fbank
    highfreq = highfreq or samplerate / 2
    signal = sigproc.preemphasis(signal, preemph)
    slen = len(signal)
    frame_len = winlen * samplerate
    frame_step = winstep * samplerate
    frame_len = int(sigproc.round_half_up(frame_len))
    frame_step = int(sigproc.round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = int(math.ceil((1.0 * slen - frame_len) / frame_step))
    padlen = int(numframes * frame_step + frame_len)
    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((signal, zeros))
    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len), (numframes, 1))
    frames = frames * win

    pspec = sigproc.magspec(frames, nfft)

    fb = get_filterbanks_alt(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    feat = np.where(feat == 0, np.finfo(float).eps, feat)  # if feat is zero, we get problems with log

    feat = np.log(feat)
    # feat = dct(feat, type=3, axis=1, norm='ortho')
    dctdct = dct(np.eye(numcep), type=3, norm='ortho')
    dctdct[0, :] *= np.sqrt(2)
    feat = np.matmul(dctdct, feat.T)

    lifter = ceplifter_coefficients(numcep, ceplifter)
    mfcc = np.matmul(np.diag(lifter), feat)
    mfcc = mfcc[1:, :]

#    if os.path.exists(save_mfcc_dir):
#        shutil.rmtree(save_mfcc_dir)
#    os.mkdir(save_mfcc_dir)

    data=[]
    num_bins = np.floor(1.0 * audio.size / sample_rate * 25).astype(int)
    for i in np.arange(0, 66):                                              #change the frame number of each video
        data.append(mfcc[:, (i*4):(i*4+35)])
    data=np.expand_dims(data,axis=4)
    np.save(os.path.join(save_mfcc_dir, 'mfcc.npy'),data)