import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def trajectoires(fichier): #crée un dictionnaire dont les clés sont les particules
    data = pd.read_csv(fichier)
    particules = {n: traj for n, traj in data.groupby('particle')}
    for n in particules:
        particules[n]["r_squared"] = (particules[n]["x"]-particules[n]["x"].iloc[0])**2 + (particules[n]["y"]-particules[n]["y"].iloc[0])**2
        particules[n]["time"] = particules[n]["frame"] - particules[n]["frame"].iloc[0] #Temps décalé pour commencer à 0
    return particules


def trace(particules, n): #trace une particule
    time = particules[n]["time"]
    r_squared = particules[n]["r_squared"]
    plt.figure(figsize=(8, 6))
    plt.plot(time, r_squared, label="Points de données", color="blue")
    plt.title(f"r² en fonction du temps pour la particule {n}")
    plt.xlabel("Temps (frame)")
    plt.ylabel("r² (px²)")
    plt.grid(True)
    plt.show()

def trace_each(particules, liste_n):
    for n in liste_n:
        if n in particules:
            trace(particules,n)

def regression(X, Y):
    mask = ~np.isnan(X) & ~np.isnan(Y)  # Crée un masque pour ignorer les NaN
    Xcorr = X[mask]
    Ycorr = Y[mask]
    slope, intercept, r_value, p_value, std_err = linregress(Xcorr,Ycorr)
    Z = Xcorr * slope + intercept
    return slope, Xcorr, Z

def MSD(particules, liste_n, frequence, trace = True, droite = True): #trace le MSD
    t_max = max([particules[n]["time"].iloc[-1] for n in liste_n])
    time = np.arange(0, t_max+1)
    msd = []
    for t in time:
        somme = 0
        nombre = 0
        for n in liste_n:
            if t in particules[n]["time"].values:
                somme += particules[n][particules[n]['time'] == t]['r_squared'].iloc[0]
                nombre += 1
        if nombre != 0:
            msd.append(somme/nombre)
        else:
            msd.append(np.nan)
    if trace:
        plt.figure(figsize=(8, 6))
        plt.scatter(time, msd, color="blue", s=5)
        plt.title(f"MSD en fonction du temps pour {len(liste_n)} particules")
        plt.xlabel("Temps (frame)")
        plt.ylabel("MSD (px²)")
        if droite:
            msd = np.array(msd)
            slope, time_corr, Z = regression(time, msd)
            plt.plot(time_corr, Z, color="red", label = f"pente = {slope*frequence*(140*10**(-9))**2:.4g} m²/s\n D = {slope*frequence*(140*10**(-9))**2/4:.4g} m²/s")
        plt.legend()
        plt.grid(True)
        plt.show()
    return msd

def Dn(particules, liste_n, frequence):
    DnList =[] #liste des D de chaque particule
    for n in liste_n:
        SD = []
        for i in range(len(particules[n])-1):
            x2, y2= particules[n]["x"].iloc[i+1], particules[n]["y"].iloc[i+1]
            x, y = particules[n]["x"].iloc[i], particules[n]["y"].iloc[i]
            dt = particules[n]["time"].iloc[i+1] - particules[n]["time"].iloc[i]
            SD.append(((x2-x)**2+(y2-y)**2)/(4*dt))
        if len(SD) > 0 :
            DnList.append(np.nanmean(SD))
    D = np.array(DnList)*frequence*(140*10**(-9))**2 #Changement d'unité
    return D, np.nanmean(D)

def trace_Dn(particules, liste_n, frequence):
    DList, D = Dn(particules, liste_n, frequence)
    plt.figure(figsize=(8, 6))
    #plt.scatter(np.arange(0, len(DList)), DList, color="blue", s = 5)
    plt.hist(DList, bins=100, edgecolor='black', alpha=0.7)
    plt.title("Histogramme de Dn")
    #plt.xlim(1e-14,5e-13)
    #plt.ylim(0, 50)
    plt.xlabel("D (m²/s)")
    plt.ylabel("nombre de particules")
    plt.show()

"""
frequence91 = 100
frequence92 = 10
particules91 = trajectoires("film91.csv")
particules92 = trajectoires("test9.2.csv")
particules6 = trajectoires("test6pierre.csv")
particules5 = trajectoires("test5.csv")
particules4 = trajectoires("test4.csv")
particulesMano = trajectoires("MANOfilm4.csv")

liste_n_92=[2,6,7,10,12,13,14,15]
liste_n_91=[6,9,21,33 ]
liste_n_6=[1,2,3,4,5,6,7,17,18,21,25,26,32,33,39,40,41,47]
liste_n_5=[32,35,90,11,6,2,84]
liste_n_4=[6,15,134,270,325,]
liste_n = [1]

#trace_each(particules4, np.arange(1,500))
_=MSD(particules91, liste_n_91, frequence91, trace = True, droite = True)

DList91, D91 = Dn(particules91, particules91, frequence91)
print(f"D91 = {D91} m²/s")

DList92, D92 = Dn(particules92, particules92, frequence92)
print(f"D92 = {D92} m²/s")

DList6, D6 = Dn(particules6, particules6, frequence92)
print(f"D6 = {np.nanmean(DList6)} m²/s")

DList5, D5 = Dn(particules5, particules5, frequence92)
print(f"D5 = {D5} m²/s")

DList4, D4 = Dn(particules4, particules4, frequence92)
print(f"D4 = {D4} m²/s")

DListM, DM = Dn(particulesMano, particulesMano, frequence92)
print(f"DM = {DM} m²/s")


#trace_Dn(particules, liste_n_92, frequence)
#trace_Dn(particules5, particules5, frequence92)
trace_Dn(particules4, particules4, frequence92)
"""
