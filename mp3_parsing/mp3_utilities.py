import os
import subprocess
from tempfile import mktemp
from scikits.audiolab import wavread, play
from scipy.signal import remez, lfilter
from pylab import *
import matplotlib.pyplot as plt

from pylab import specgram, show
import scipy.signal as signal
from scikits.talkbox.features.mfcc import mfcc, trfbank

def mp3ToDFT(mp3filename):
    # convert mp3, read wav
    wname = mktemp('.wav')
    FNULL = open(os.devnull, 'w')
    subprocess.call(['avconv', '-i', mp3filename, "-ss", "30", "-t", "30", wname], stdout=FNULL, stderr=subprocess.STDOUT)
    sig, fs, enc = wavread(wname)

    #todo: remove wav file
    #todo: convert to mono, averaging sig[:,0] + sig[;,1]
    os.unlink(wname)

    mfcc_feat = mfcc(sig, fs=fs)[1]

    return mfcc_feat