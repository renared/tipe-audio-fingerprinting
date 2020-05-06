''' Le code qui suit est inachevé : il permet néanmoins de calculer une analyse multirésolution (AMR) d'un extrait d'une manière similaire au calcul du spectrogramme dans les programmes précédents, et de la représenter graphiquement ; il permet aussi de rechercher les pics d'amplitude dans une AMR mais le format sous lequel on les obtient doit être corrigé. '''

import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy import signal
import numpy as np
import sqlite3
import os
import pywt

CHUNK_SIZE = 8192
CHUNK_OVERLAP = 0.25
PEAK_MIN_DIST = 0.1
PEAK_THRESHOLD = 0.2
PEAK_NPEAKS = 4

def downsample(s, factor):  # sous-échantillonage
    return [s[i] for i in range(0,len(s),factor)]
    
def mix_channels(s):    # conversion stéréo -> mono
    s = np.array(s).transpose()
    return (s[0]+s[1])/2
    
def normalize(s):   # normalise une liste
    l = np.array(s)
    minamp = np.amin(l)
    maxamp = np.amax(l)
    return 2*l/(maxamp-minamp)
    
def chunks(s, sampFreq, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):    # AMR "par blocs" : analyses multirésolutions effectuées sur des subdivisions régulières consécutives du signal s qui se chevauchent
    n_chunks = 0
    chunks = []
    t = []
    a = 0
    amax = len(s)-size
    s = normalize(s)
    while (a <= amax):
        b = a+size
        t.append(a/sampFreq)
        chunks.append( pywt.wavedec(s[a:b], "coif1") )
        n_chunks+=1
        a = b - int(overlap*size)
    return chunks

def peaks(chunks):  # recherche des pics
    pks = []
    for j in range(len(chunks[0])):
        allscalepeaks = []
        for i in range(len(chunks)):
            scalepeaks = []
            sortedscale = sorted([(l,chunks[i][j][l]) for l in range(len(chunks[i][j]))],key=lambda x:x[1], reverse=True)
            k = int(CHUNK_OVERLAP*len(sortedscale)/2)
            while k <= (1-CHUNK_OVERLAP/2)*len(sortedscale) and len(scalepeaks)<PEAK_NPEAKS:
                l,x = sortedscale[k][0],sortedscale[k][1]**2
                if x >= PEAK_THRESHOLD:
                    ok = True
                    for l1,x1 in scalepeaks:
                        if abs(l-l1)/len(sortedscale) < PEAK_MIN_DIST:
                            ok = False
                            break
                    if ok : scalepeaks.append((l,x))
                k+=1
            scalepeaksglobal = [ (x[0]+(i+1-CHUNK_OVERLAP/2)*len(sortedscale), x[1]) for x in scalepeaks ]
            allscalepeaks+=scalepeaksglobal
        pks.append(allscalepeaks)
    return pks
            
def graph1(chunk):  # représente graphiquement une AMR
    l = chunk
    fig, ax = plt.subplots(len(l), sharex=True)
    for j in range(len(l)):
        ax[j].plot(np.linspace(0,1,num=len(l[j])),l[j])
        
def graph2(chunks): # représente graphiquement une AMR "par blocs"
    fig, ax = plt.subplots(len(chunks[0]), sharex=True)
    for j in range(len(chunks[0])):
        for i in range(len(chunks)):
            l = chunks[i][j]
            a = int(CHUNK_OVERLAP*len(l)/2)
            b = int((1-CHUNK_OVERLAP/2)*len(l))
            ax[j].plot(np.linspace(i,i+1,num=b-a),l[a:b])
            
def graph3(chunks): # représente graphiquement une AMR "par blocs" ainsi que les pics d'amplitudes détectés
    pks = peaks(chunks)
    fig, ax = plt.subplots(len(chunks[0]), sharex=True)
    for j in range(len(chunks[0])):
        for i in range(len(chunks)):
            l = chunks[i][j]
            a = int(CHUNK_OVERLAP*len(l)/2)
            b = int((1-CHUNK_OVERLAP/2)*len(l))
            ax[j].plot(np.linspace(i,i+1,num=b-a),l[a:b])
        for x,y in pks[j]:
            ax[j].scatter(x/len(chunks[0][j]),y**.5)
    
    

## TEST
sampFreq, snd = wav.read('C:\\TIPE\\anthem_frag.wav')
sampFreq //= 10
snd = downsample(snd,10)
snd0 = mix_channels(snd)
sxx = chunks(snd0, sampFreq)
pks = peaks(sxx)
