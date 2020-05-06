from scipy.io import wavfile as wav
import scipy.signal as signal
from math import *
import numpy as np
from time import time, sleep
import matplotlib.pyplot as plt
import sqlite3
import os
from random import random

DBfile = "C:\\TIPE\\DB\\audio_fingerprints12.db"
conn = sqlite3.connect(DBfile)
cur = conn.cursor()

DOWNSAMPLING_FACTOR = 6 # 44100 Hz -> 7350 Hz
FFTSIZE = 512   # 70 ms
OVERLAP = 0.5

PEAK_MIN_FREQ = 300.0   # après D3 
PEAK_MAX_FREQ = 1500.0  # avant G6
PEAK_THRESHOLD = 0.2
PEAK_ALPHA_HEIGHT_100HZ = 10.0
PEAK_ALPHA_WIDTH = 0.1
PEAK_FILTER = True

HASH_NPAIRS = 2
HASH_COH_WIDTH = 1.0
HASH_COH_HEIGHT_100HZ = 50.0
HASH_P_A = 1.0
HASH_P_B = 1.0

DEBUG_TIME_GLOBAL_IMPORT = True
DEBUG_TIME_GLOBAL_MATCHING = True

def downsample(s, factor):  # sous-échantillonnage
    return [s[i] for i in range(0,len(s),factor)]

def mix_channels(s):    # conversion stéréo -> mono
    s = np.array(s).transpose()
    return (s[0]+s[1])/2
    
def normalize_2d(l):    # normalise une liste à deux dimensions
    l = np.array(l)
    minamp = np.amin(l)
    maxamp = np.amax(l)
    return (l-minamp)/(maxamp-minamp)
    
def spectrogram(s, sampFreq, fftsize=8192, overlap=0.25):   # spectrogramme
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
    
    
def peaks(t,f,Sxx): # recherche des pics d'amplitudes
    w = int(PEAK_ALPHA_WIDTH/t[1])
    maxima = []
    j0 = ceil(PEAK_MIN_FREQ/f[1])
    j1 = floor(PEAK_MAX_FREQ/f[1])+1
    for i in range(w,len(Sxx)-w):
        for j in range(j0,j1):
            x = Sxx[i][j]
            h = int(PEAK_ALPHA_HEIGHT_100HZ*j/100.0)
            
            if x >= PEAK_THRESHOLD :    # seuil d'amplitude
                if i-w>=0 and i+w<len(Sxx) and j-h>=0 and j+h<len(Sxx[0]) : # contre les effets de bord
                    ok = True
                    for u in range(i-w,i+w+1):  # test si tous les points dans une zone autour de (i,j) sont de plus faible amplitude que le point (i,j)
                        if not(ok) : break
                        for v in range(j-h,j+h+1):
                            y = Sxx[u][v]
                            if y > x :
                                ok = False
                                break

                    if ok : 
                        maxima.append((i,j,x))
    if PEAK_FILTER: # permet de ne conserver que les pics d'amplitude supérieure à l'amplitude moyenne de tous les pics
        m = sum([x[2] for x in maxima])/len(maxima)
        return [x for x in maxima if x[2]>=m]
    return maxima

    
    
def hashpeaks(t,f,pks,hashoutput=True): # appairage des pics
    sortedpeaks = sorted(pks,key=lambda x: x[0])
    hashes = []
    npairs = HASH_NPAIRS
    for i in range(len(sortedpeaks)-1):
        j = i+1
        pairs = []
        coh_width = HASH_COH_WIDTH
        coh_height_100hz = HASH_COH_HEIGHT_100HZ
        coh_height = coh_height_100hz*f[sortedpeaks[i][1]]/100
        
        strongestpeak_x,strongestpeak_k = 0,0
        
        while j < len(sortedpeaks) and t[sortedpeaks[j][0]]-t[sortedpeaks[i][0]] <= coh_width :
            if sortedpeaks[j][0]!=sortedpeaks[i][0]:
                df = f[sortedpeaks[j][1]]-f[sortedpeaks[i][1]]
                
                if abs(df) <= coh_height/2 :
                    t0 = sortedpeaks[i][0]
                    t1 = sortedpeaks[j][0]
                    f0 = sortedpeaks[i][1]
                    f1 = sortedpeaks[j][1]
                    x = sortedpeaks[j][2]
                    
                    d = sqrt( ((1/coh_width)*(t[t1]-t[t0]))**2 + ((2/coh_height)*(f[f1]-f[f0]))**2 + (sortedpeaks[j][2]-sortedpeaks[i][2])**2 ) # distance euclidienne dans l'espace (temps, fréquence, amplitude)
                    
                    if x > strongestpeak_x : strongestpeak_x,strongestpeak_k = x,len(pairs)
                    data = (t0,int(str(hash((t1-t0,f0,f1)))[-8:])) if hashoutput else (t0,t1-t0,f0,f1) # hachage des données
                    pairs.append((d, data))
            j+=1
        if len(pairs)!=0: 
            if random()<HASH_P_A : hashes.append(pairs.pop(strongestpeak_k)[1]) # peak le plus fort, tout en le retirant de la liste
            sortedpairs = sorted(pairs,key=lambda x: x[0],reverse=False) # on trie distance croissant
            for i in range(min(len(pairs),npairs)):
                if random()<HASH_P_B : hashes.append(sortedpairs[i][1])
    return hashes

    
def intodb(path):   # importation dans la BDD
    t1 = time()
    m,n = 0,0
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        for fname in fileList:
            if fname.endswith(".wav"):
                cur.execute("SELECT COUNT(*) FROM songs WHERE filename=?", (fname,))
                if list(cur)[0][0] > 0 :
                    print(os.path.join(dirName, fname),"déjà dans la BDD")
                    m+=1
                else :
                    print("Importation de",os.path.join(dirName, fname))
                    sampFreq, snd = wav.read(os.path.join(dirName, fname))
                    snd0 = downsample(mix_channels(snd),DOWNSAMPLING_FACTOR)
                    sampFreq0 = sampFreq//DOWNSAMPLING_FACTOR
                    t,f,Sxx = spectrogram(snd0, sampFreq0, FFTSIZE, OVERLAP)
                    pks = peaks(t,f,Sxx)
                    hts = hashpeaks(t,f,pks)
                    data = [(fname,)]
                    for item in data:
                        cur.execute("INSERT INTO songs(filename) VALUES(?)", item)
                    cur.execute("SELECT id FROM songs WHERE filename = ?", (fname,))
                    id = list(cur)[0][0]
                    for item in [(id, ht[0], ht[1]) for ht in hts]:
                        cur.execute("INSERT INTO hashes(idSong,t,h) VALUES(?,?,?)", item)
                    conn.commit()
                    n+=1
    t2 = time()
    print(m,"musiques étaient déjà importées dans la BDD")
    print(n,"musiques ont bien été importées dans la BDD")
    if DEBUG_TIME_GLOBAL_IMPORT and n>0 : print("=== Durée totale de l'importation :",t2-t1," === Durée moyenne :",(t2-t1)/n,"par fichier")
    
def identify(file,dontdelete=False):    # identification
    t1 = time()
    sampFreq, snd = wav.read(file)
    snd0 = downsample(mix_channels(snd),DOWNSAMPLING_FACTOR)
    sampFreq0 = sampFreq//DOWNSAMPLING_FACTOR
    t,f,Sxx = spectrogram(snd0, sampFreq0, FFTSIZE, OVERLAP)
    pks = peaks(t,f,Sxx)
    hts = hashpeaks(t,f,pks)
    for item in [(0, ht[0], ht[1]) for ht in hts]:
        cur.execute("INSERT INTO hashes(idSong,t,h) VALUES(?,?,?)", item)
    conn.commit()
    cur.execute("SELECT songs.filename, b.t-a.t, COUNT(*) FROM hashes a, hashes b JOIN songs ON songs.id=b.idSong WHERE a.idSong=0 AND b.idSong!=0 AND b.t >= a.t AND b.h=a.h GROUP BY b.t-a.t, b.idSong ORDER BY COUNT(*) DESC LIMIT 1")   # requête permettant l'identification
    print(list(cur))
    if not(dontdelete):
        cur.execute("DELETE FROM hashes WHERE idSong=0")    # suppression des données relatives à l'extrait
        conn.commit()
        while True: # on attend que les données de l'extrait soient effectivement supprimées
            cur.execute("SELECT EXISTS (SELECT id FROM hashes WHERE idSong=0)")
            if list(cur)[0][0]==0: break
            else: sleep(0.5)
    if DEBUG_TIME_GLOBAL_MATCHING : print("=== Durée totale du matching :",time()-t1)
    




##  fonctions diverses

    
def deleteSong0():  # supprimer les données relatives à l'extrait manuellement
    cur.execute("DELETE FROM hashes WHERE idSong=0")
    conn.commit()

def clearDB(): # vide la base de données
    cur.execute("DELETE FROM songs")
    cur.execute("DELETE FROM hashes")
    cur.execute("DELETE FROM sqlite_sequence")
    conn.commit()
    
def convertArrays(t,f,Sxx): # convertit les listes relatives au spectrogramme pour matplotlib
    X,Y=np.array(t),np.array(f)
    Z = np.array(Sxx).transpose()
    return X,Y,Z
    
def graph(t,f,Sxx, xmin, xmax, ymin, ymax, shading = False):    # représente graphiquement le spectrogramme
    X,Y,Z = convertArrays(t,f,Sxx)

    if shading : plt.pcolormesh(X,Y,Z,cmap='Greys', shading="gouraud")
    else : plt.pcolormesh(X,Y,Z,cmap='Greys')
    #plt.colorbar()
    plt.axis([xmin,xmax,ymin, ymax])
    plt.semilogy(nonposy="mask")
    plt.show()
    
def graphhashpeaks(t,f,pks):    # permet de représenter graphiquement les paires (copie modifiée de hashpeaks)
    sortedpeaks = sorted(pks,key=lambda x: x[0])
    hashes = []
    npairs = HASH_NPAIRS
    for i in range(len(sortedpeaks)-1):
        j = i+1
        pairs = []
        coh_width = HASH_COH_WIDTH
        coh_height_100hz = HASH_COH_HEIGHT_100HZ
        coh_height = coh_height_100hz*f[sortedpeaks[i][1]]/100
        
        strongestpeak_x,strongestpeak_k = 0,0
        
        while j < len(sortedpeaks) and t[sortedpeaks[j][0]]-t[sortedpeaks[i][0]] <= coh_width :
            if sortedpeaks[j][0]!=sortedpeaks[i][0]:
                df = f[sortedpeaks[j][1]]-f[sortedpeaks[i][1]]
                
                if abs(df) <= coh_height/2 :
                    t0 = sortedpeaks[i][0]
                    t1 = sortedpeaks[j][0]
                    f0 = sortedpeaks[i][1]
                    f1 = sortedpeaks[j][1]
                    x = sortedpeaks[j][2]
                    d = abs(t1-t0)+abs(f1-f0)
                    if x > strongestpeak_x : strongestpeak_x,strongestpeak_k = x,len(pairs)
                    pairs.append((d, (t[t0],t[t1],f[f0],f[f1])) )
            j+=1
        if len(pairs)!=0: 
            hashes.append(pairs.pop(strongestpeak_k)[1])
            sortedpairs = sorted(pairs,key=lambda x: x[0],reverse=False)
            for i in range(min(len(pairs),npairs)):
                if HASH_P>=1 or random()<HASH_P : hashes.append(sortedpairs[i][1])
    for pair in hashes:
        plt.plot( [ pair[0], pair[1] ], [ pair[2], pair[3] ], c=(random(),random(),random()), alpha=0.6)
    print(len(hashes), "pairs")
    return






## TESTS

if False :  # test de différentes valeures de "convolution", obsolète
    sampFreq, snd = wav.read("C:\\TIPE\\zeddspectrum.wav")
    snd0 = downsample(mix_channels(snd),DOWNSAMPLING_FACTOR)
    sampFreq0 = sampFreq//DOWNSAMPLING_FACTOR
    t,f,Sxx = spectrogram(snd0, sampFreq0, FFTSIZE, OVERLAP)
    plt.ioff()
    for i in range(1,6):
        for j in range(1,6):
            m = peaks(t,f,Sxx,h=i,w=j)
            graph(t,f,Sxx,0,t[-1],50,sampFreq0/2, shading=False)
            plt.scatter([t[x[0]] for x in m], [f[x[1]] for x in m], s=5)
            plt.savefig("C:\\TIPE\\figures\\reprise mai 2018\\conv2\\conv "+str(i)+" "+str(j)+".png",dpi=1000)
            plt.close()

if False :   # test pics
    sampFreq, snd = wav.read("C:\\TIPE\\zeddspectrum.wav")
    snd0 = downsample(mix_channels(snd),DOWNSAMPLING_FACTOR)
    sampFreq0 = sampFreq//DOWNSAMPLING_FACTOR
    t,f,Sxx = spectrogram(snd0, sampFreq0, FFTSIZE, OVERLAP)
    m = peaks(t,f,Sxx)
    graph(t,f,Sxx,0,t[-1],50,sampFreq0/2, shading=False)
    plt.scatter([t[x[0]] for x in m], [f[x[1]] for x in m], s=22)
    
if False :   # test hashpeaks
    sampFreq, snd = wav.read("C:\\TIPE\\zeddspectrum.wav")
    snd0 = downsample(mix_channels(snd),DOWNSAMPLING_FACTOR)
    sampFreq0 = sampFreq//DOWNSAMPLING_FACTOR
    t,f,Sxx = spectrogram(snd0, sampFreq0, FFTSIZE, OVERLAP)
    pks = peaks(t,f,Sxx)
    graph(t,f,Sxx,0,t[-1],50,sampFreq0/2, shading=False)
    plt.scatter([t[x[0]] for x in pks], [f[x[1]] for x in pks], s=22)
    print(len(pks), "peaks")
    graphhashpeaks(t,f,pks)
    
if False :   # test intersection de deux hashpeaks
    sampFreq, snd = wav.read("C:\\TIPE\\stressedout_voiceoverb.wav")
    snd0 = downsample(mix_channels(snd),DOWNSAMPLING_FACTOR)
    sampFreq0 = sampFreq//DOWNSAMPLING_FACTOR
    t,f,Sxx = spectrogram(snd0, sampFreq0, FFTSIZE, OVERLAP)
    pks = peaks(t,f,Sxx)
    graph(t,f,Sxx,0,t[-1],50,sampFreq0/2, shading=False)
    plt.scatter([t[x[0]] for x in pks], [f[x[1]] for x in pks], s=35)
    print(len(pks), "peaks")
    hts = hashpeaks(t,f,pks,hashoutput=False)
    
    sampFreq2, snd2 = wav.read("C:\\TIPE\\stressedout_frag_cam.wav")
    snd02 = downsample(mix_channels(snd2),DOWNSAMPLING_FACTOR)
    sampFreq02 = sampFreq2//DOWNSAMPLING_FACTOR
    t2,f2,Sxx2 = spectrogram(snd02, sampFreq02, FFTSIZE, OVERLAP)
    pks2 = peaks(t2,f2,Sxx2)
    plt.scatter([t2[x[0]]+0.00001 for x in pks2], [f2[x[1]]+0.00001 for x in pks2], s=15, c="green")
    print(len(pks2), "peaks")
    hts2 = hashpeaks(t2,f2,pks2,hashoutput=False)
    
    htsinter = set(hts) & set(hts2)
    for pair in htsinter:
        plt.plot( [ t[pair[0]], t[pair[1]+pair[0]] ], [ f[pair[2]], f[pair[3]] ], c=(random(),random(),random()), alpha=0.6)
    print(len(htsinter),"paires en commun")
    
if False :   # test intersection de deux hashpeaks (v2)
    sampFreq, snd = wav.read("C:\\TIPE\\stressedout_voiceoverb.wav")
    snd0 = downsample(mix_channels(snd),DOWNSAMPLING_FACTOR)
    sampFreq0 = sampFreq//DOWNSAMPLING_FACTOR
    t,f,Sxx = spectrogram(snd0, sampFreq0, FFTSIZE, OVERLAP)
    pks = peaks(t,f,Sxx)
    graph(t,f,Sxx,0,t[-1],50,sampFreq0/2, shading=False)
    plt.scatter([t[x[0]] for x in pks], [f[x[1]] for x in pks], s=35)
    print(len(pks), "peaks")
    hts = hashpeaks(t,f,pks, hashoutput=False)
    print(len(hts),"hashes")
    
    sampFreq2, snd2 = wav.read("C:\\TIPE\\audiotipe\\twenty one pilots - Stressed Out.wav")
    T0_ = 1*60+20.00
    snd02 = downsample(mix_channels(snd2),DOWNSAMPLING_FACTOR)
    sampFreq02 = sampFreq2//DOWNSAMPLING_FACTOR
    t2,f2,Sxx2 = spectrogram(snd02, sampFreq02, FFTSIZE, OVERLAP)
    T0 = int(round(T0_/t2[1]))
    pks2_ = peaks(t2,f2,Sxx2)
    hts2_ = hashpeaks(t2,f2,pks2_,hashoutput=False)
    pks2 = [(pks2_[i][0]-T0,pks2_[i][1],pks2_[i][2]) for i in range(len(pks2_)) if pks2_[i][0]-T0>=0]
    plt.scatter([t2[x[0]]+0.001 for x in pks2], [f2[x[1]]+0.001 for x in pks2], s=15, c="green")
    hts2 = [(x[0]-T0,x[1],x[2],x[3]) for x in hts2_]
    print(len(pks2_), "peaks")
    print(len(hts2_),"hashes")
    
    htsinter = set(hts) & set(hts2)
    for pair in htsinter:
        plt.plot( [ t[pair[0]], t[pair[1]+pair[0]] ], [ f[pair[2]], f[pair[3]] ], c=(random(),random(),random()), alpha=0.6)
    print(len(htsinter),"paires en commun")
    
    
    

if False :
    sampFreq, snd = wav.read("C:\\TIPE\\zeddspectrum.wav")
    snd0 = downsample(mix_channels(snd),6)
    sampFreq0 = sampFreq//6
    t,f,Sxx = spectrogram(snd0,sampFreq0,512,0.50)
    #t,f,Sxx = spectrogram(mix_channels(snd),sampFreq,8192,0.5)
    #testpeaks2(t,f,Sxx,1,100,200.0,4000, threshold=0.5, npoints=int(10*t[-1]))
    pks = peaks(t,f,Sxx,1,100,200.0,4000, threshold=0.5, npoints=int(15*t[-1]))
    graph(t,f,Sxx,0,t[-1],50,sampFreq0/2, shading=True)
    plt.scatter([t[p[0]] for p in pks], [f[p[1]] for p in pks], s=30, c='orange')
    print(len(pks), "peaks")
    hashes = hashpeaks(t,f,pks,0.1,5,1.2,1000)
    
if False :
    sampFreq, snd = wav.read("C:\\TIPE\\zeddspectrum.wav")
    snd0 = downsample(mix_channels(snd),DOWNSAMPLING_FACTOR)
    sampFreq0 = sampFreq//DOWNSAMPLING_FACTOR
    t,f,Sxx = spectrogram(snd0, sampFreq0, FFTSIZE, OVERLAP)
    pks = peaks(t,f,Sxx,PEAK_TILE_WIDTH, PEAK_TILE_HEIGHT_100HZ, PEAK_MIN_FREQ, PEAK_MAX_FREQ, threshold=PEAK_THRESHOLD, npoints=int(PEAK_MAX_POINTS_AVG_PER_SECOND*t[-1]))
    graph(t,f,Sxx,0,t[-1],50,sampFreq0/2, shading=False)
    hts = graphhashpeaks(t,f,pks,HASH_TILE_WIDTH, HASH_TILE_HEIGHT_100HZ, HASH_COH_WIDTH, HASH_COH_HEIGHT_100HZ)
    plt.scatter([t[p[0]] for p in pks], [f[p[1]] for p in pks], s=30, c='orange')
    print(len(pks), "peaks")
    
if False :  # test d'identification d'un échantillon d'extraits
    liste_fichiers = ["C:\\TIPE\\zeddspectrum.wav",
                      #"C:\\TIPE\\zeddspectrum_bruit0.wav",
                      "C:\\TIPE\\zeddspectrum_bruit.wav",
                      "C:\\TIPE\\zeddspectrum_cam.wav",
                      "C:\\TIPE\\stressedout_frag.wav",
                      "C:\\TIPE\\stressedout_frag_cam.wav",
                      #"C:\\TIPE\\stressedout_frag_cam2.wav",
                      "C:\\TIPE\\stressedout_voiceover.wav",
                      "C:\\TIPE\\stressedout_voiceover1.wav",
                      "C:\\TIPE\\stressedout_voiceover2.wav",
                      "C:\\TIPE\\anthem_frag.wav",
                      "C:\\TIPE\\anthem_frag_cam.wav"
                      ]
    for fichier in liste_fichiers :
        print("Pour",fichier,":")
        identify(fichier)
        print("\n")