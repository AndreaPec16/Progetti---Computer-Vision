import cv2
import numpy as np
import pickle
import os
from sklearn.preprocessing import normalize

def predici_nuova_immagine(path_img, file_vocabolario, file_modello, k):
    """
    Esegue la pipeline di inferenza su una singola immagine.
    """
    print(f"Analisi immagine: {path_img}")

    # Caricamento degli elementi necessari all'inferenza 
    try:
        with open(file_vocabolario, "rb") as f:
            kmeans = pickle.load(f)
        with open(file_modello, "rb") as f:
            classificatore = pickle.load(f)
    except FileNotFoundError:
        print("File del modello o del vocabolario non trovati")
        return None

    #  Caricamento immagine 
    img = cv2.imread(path_img)
    if img is None:
        print("Immagine non trovata")
        return None
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Estrazione descrittori locali
    sift = cv2.SIFT_create()
    _, descrittori = sift.detectAndCompute(img_gray, None)

    if descrittori is None:
        print("Nessun keypoint rilevato nell'immagine")
        return None

    # Assegnare i descrittori alle visual words del vocabolario appreso 
    descrittori = normalize(descrittori, norm="l2")
    words = kmeans.predict(descrittori)
    
    # Costruzione istogramma BoW dell'immagine 
    hist, _ = np.histogram(words, bins=np.arange(k + 1))

    # Normalizzazione 
    hist_normalizzato = normalize(hist.reshape(1, -1), norm="l1")

    # Predizione classe
    predizione = classificatore.predict(hist_normalizzato)

    print(f"\n Classe predetta: {predizione[0]}")
    return predizione[0]

if __name__ == "__main__":
    print("Inizio processo inferenza")

    k_vincente = 500
    path_vocabolario = f"vocabolario_k{k_vincente}.pkl"
    path_modello = "miglior_classificatore.pkl"

    immagine_test = input("\nInserisci il percorso dell'immagine da classificare : ")
    

    if os.path.exists(immagine_test):
        predici_nuova_immagine(immagine_test, path_vocabolario, path_modello, k_vincente)
        print("\nInferenza completata")
    else:
        print(f"File '{immagine_test}' non trovato")
