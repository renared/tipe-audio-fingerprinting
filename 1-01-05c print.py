''' Le code qui suit constitue une première approche effectuée en mai-juin 2017. '''

import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
import scipy.signal as signal
from math import *
import numpy as np
import os
import sqlite3
from time import time


STEREO = True
#RANGES = [(0,80),(80,200),(200,500),(500,1200),(1200,4000),(4000,10000)]
#RANGES = [(20, 80), (80, 150), (150, 300), (300, 500), (500, 800), (800, 1500)]
#RANGES = [(60, 150), (150, 300), (300, 550), (550, 900), (900, 1300), (1300, 2000)]
RANGES = [(80, 120), (120, 180), (180, 300), (300, 500), (500, 880),(880,1549)]
FFTSIZE = 8192

DBfile = "E:\\DB\\audio_fingerprints.sqlite"
conn = sqlite3.connect(DBfile)
cur = conn.cursor()

def normalize_2d(l):    # normalise une liste à deux dimensions
    l = np.array(l)
    minamp = np.amin(l)
    maxamp = np.amax(l)
    return (l-minamp)/(maxamp-minamp)

def spectrogram(s, sampFreq, fftsize=8192, overlap=0.50):   # spectrogramme
    chunklen = fftsize/(1+overlap)
    n_chunks = 0
    window = signal.blackmanharris(fftsize)
    chunks = []
    t = []
    a = 0
    amax = len(s)-fftsize
    while (a <= amax):
        b = a+fftsize
        t.append(a/sampFreq)
        chunks.append(20*np.log10( np.maximum([0.004 for i in range(fftsize//2+1)],np.abs(np.fft.rfft(window*s[a:b])))))
        n_chunks+=1
        a = b - int(overlap*fftsize)
    f = np.fft.rfftfreq(fftsize,1/sampFreq)
    chunks = normalize_2d(chunks)
    return np.array(t),np.array(f),np.array(chunks)
    
    return t, f, chunks
    
def convertArrays(t,f,Sxx): # conversion de listes pour matplotlib
    X,Y=np.array(t),np.array(f)
    Z = np.array(Sxx).transpose()
    return X,Y,Z
    
def graph(t,f,Sxx, xmin, xmax, ymin, ymax, shading = False):    # représentation graphique du spectrogramme
    X,Y,Z = convertArrays(t,f,Sxx)
    if shading : plt.pcolormesh(X,Y,Z, shading="gouraud")
    else : plt.pcolormesh(X,Y,Z)
    plt.colorbar()
    plt.axis([xmin,xmax,ymin, ymax])
    plt.show()
    
def freqrtoindexr(f,f0,f1): # conversion
    imin = 0
    for i in range(1,len(f)):
        if f[i] > f0 : 
            imin = i-1
            break
    for i in range(imin+1,len(f)):
        if f[i] > f1:
            imax = i-1
            return imin,imax
    return
    
def intodb(path):   # importation dans la base de données
    m,n = 0,0
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        for fname in fileList:
            if fname.endswith(".wav"):
                cur.execute("SELECT COUNT(*) FROM songs WHERE filename=?", (fname,))
                if list(cur)[0][0] > 0 :
                    print(os.path.join(dirName, fname),"déjà dans la BD")
                    m+=1
                else :
                    print("Importation de",os.path.join(dirName, fname))
                    sampFreq, snd = wav.read(os.path.join(dirName, fname))
                    snd0 = snd[:,0]
                    t,f,Sxx = spectrogram(snd0, sampFreq, FFTSIZE)
                    pks = peaks(t,f,Sxx, RANGES)
                    data = [(fname,)]
                    for item in data:
                        cur.execute("INSERT INTO songs(filename) VALUES(?)", item)
                    cur.execute("SELECT id FROM songs WHERE filename = ?", (fname,))
                    id = list(cur)[0][0]
                    for item in [(id, i, pks[i][0],pks[i][1],pks[i][2],pks[i][3],pks[i][4],pks[i][5]) for i in range(len(pks))]:
                        cur.execute("INSERT INTO points(idSong,t,f1,f2,f3,f4,f5,f6) VALUES(?,?,?,?,?,?,?,?)", item)
                    conn.commit()
                    n+=1
    print(m,"musiques étaient déjà importées dans la BD")
    print(n,"musiques ont bien été importées dans la BD")
                
def identify(s):    # identification du signal s
    t,f,Sxx = spectrogram(snd0, sampFreq, FFTSIZE)
    pa = peaks(t,f,Sxx, RANGES )
    cur.execute("SELECT id FROM songs")
    songs_id = [x[0] for x in list(cur)]
    min_d, min_id = float("inf"), 0
    for id in songs_id:
        cur.execute("SELECT f1,f2,f3,f4,f5,f6 FROM points WHERE idSong=? ORDER BY t", (id,))
        p = list(cur)
        d = difference(pa, p)
        print("idSong =",id," différence d =",d)
        if d < min_d : min_d, min_id = d, id
    return min_id
    
def peaks(t,f,Sxx,RANGES2): # pics
    THRESHOLD = 0.66
    S0 = np.array(Sxx).transpose()
    pks = []
    for f0, f1 in RANGES2:
        fa,fb = freqrtoindexr(f,f0,f1)
        l = []
        for i in range(1,len(S0[0])-1): 
            amax = np.argmax(Sxx[i][fa:fb+1])+fa
            if S0[amax][i] >= THRESHOLD : l.append(f[amax])
            else : l.append(-1)
        pks.append(l)
    pks = np.array(pks).transpose()
    return pks
    
def sim(u,v):   # Fonction de similarité : renvoie 1 si u = v, renvoie 0 si la différence entre u et v est maximale
    if len(u)!=len(v) and len(u)!=len(RANGES) : return
    p = 1
    nombreDeTrous = 0
    kmin = float("inf")
    for i in range(len(u)):
        if u[i]>0 and v[i]>0 :
            k = 1-abs(u[i]-v[i])/(RANGES[i][1]-RANGES[i][0])
            p *= k
            if k < kmin : kmin = k
        else : 
            nombreDeTrous += 1   
    if kmin == float("inf") : return 0
    p *= kmin**nombreDeTrous
    return p

def difference(tfa, tf):    # différence entre deux nuages de pics (de mêmes dimensions)
    if len(tfa[0]) != len(tf[0]) : return
    nf = len(tfa[0])
    dura = len(tfa)
    dur = len(tf)
    min_s, min_t0 = float("inf"), 0
    
    nbPas = 1  # pas, expérimental
    
    for t0 in range(0,dur-dura,nbPas):
        s = 0
        for t in range(0,dura):
            u = [tf[t0+t][i] for i in range(nf)]
            v = [tfa[t][i] for i in range(nf)]
            s+=1-sim(u,v)
        if s < min_s : min_s, min_t0 = s, t0
    print("origine temporelle la plus probable :",min_t0*FFTSIZE/sampFreq,"secondes")
    return min_s
    

## TESTS
    
def test2():    # test graph
    sampFreq, snd = wav.read("E:\\stressedout_voiceover1.wav")
    snd0 = snd[:,0] if STEREO else snd  # stereo -> mono (left channel)
    t,f,Sxx = spectrum(snd0, sampFreq, 4096)
    p = peaks(t,f,Sxx, [(20, 80), (80, 150), (150, 300), (300, 500), (500, 800), (800, 1500)] )
    graph(t,f,Sxx, 0, 10, 0, 3000, True)
    plt.scatter(*zip(*p), s=4)
    
def test3():    # test identification
    sampFreq, snd = wav.read("E:\\stressedout_voiceover1.wav")
    snd0 = snd[:,0] if STEREO else snd  # stereo -> mono (left channel)
    identifiedid = identify(snd0)
    cur.execute("SELECT filename FROM songs WHERE id=?",(identifiedid,))
    print(list(cur)[0][0])
    
def test3bis(): # test importation
    start = time()
    intodb("D:\\Dossier personnel\\Desktop\\audiotipe2")
    lap = time()
    print("L'importation a pris",lap-start,"secondes")
    test3()
    end = time()
    print("Importation :", lap-start,"s ; Identification :", end-lap,"s ; Total :", end-start,"s")