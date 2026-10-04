import cv2
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle
from sklearn.preprocessing import normalize

def get_image_paths(dir):
    """
    Scansiona la directory fornita e restituisce una lista 
    con i percorsi assoluti di tutte le immagini .jpg presenti.
    """
    paths = []
    for r, d, files in os.walk(dir):
        for file in files:
            if file.endswith(".jpg"):
                full_path  = os.path.join(r, file)
                paths.append(full_path)
    return paths


def estrazione_descrittori(paths, max_descrittori):
    """
    Utilizza SIFT per estrarre i descrittori da un insieme di immagini.
    """
    sift = cv2.SIFT_create()
    tutti_descrittori = []
    # Calcoliamo quanti descrittori massimi prendere per ogni singola immagine
    max_descrittori_img = max_descrittori// len(paths)

    for path in paths:
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        
        keypoints, descrittori = sift.detectAndCompute(img_gray, None)
        if descrittori is not None:
            # Se l'immagine ha generato troppi descrittori, ne selezioniamo un sottoinsieme casuale
            if descrittori.shape[0] > max_descrittori_img:
                id = np.random.choice(descrittori.shape[0], max_descrittori_img, replace = False)
                descrittori = descrittori[id, :]
            tutti_descrittori.append(descrittori)
    matrice_descrittori = np.vstack(tutti_descrittori)
    matrice_descrittori = normalize(matrice_descrittori, norm='l2')

    return matrice_descrittori

def genera_vocabolari_multipli(cartella_dataset, valori_k, max_descr):
    """
    Genera e salva su disco i vocabolari per diversi valori di K.
    """
    matrice_globale = None

    for k in valori_k:
        nome_file = f"vocabolario_k{k}.pkl"
        
        if os.path.exists(nome_file):
            print(f"Vocabolario {nome_file} già esistente")
            continue 
            
        if matrice_globale is None:
            print("Estrazione descrittori globale")
            paths = get_image_paths(cartella_dataset)
            matrice_globale = estrazione_descrittori(paths, max_descrittori=max_descr)
            
            if matrice_globale is None:
                print("Errore estrazione descrittori")
                return
        # Utilizziamo MiniBatchKMeans per questioni di efficienza computazionale su dataset di grandi dimensioni
        print(f"Addestramento Kmeans con K = {k}") 
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=42, n_init="auto")
        kmeans.fit(matrice_globale)
        
        with open(nome_file, "wb") as f:
            pickle.dump(kmeans, f)
        print(f"Vocabolario salvato: {nome_file}")

if __name__ == "__main__":
    cartella_aid = "./AID" 

    print("Inizio generazione vocabolari")
    genera_vocabolari_multipli(cartella_aid, valori_k=[50, 100, 500], max_descr=100000)
    print("Generazione completata")