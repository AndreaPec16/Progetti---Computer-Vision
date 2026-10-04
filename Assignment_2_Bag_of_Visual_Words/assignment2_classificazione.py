import cv2
import os
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

def get_image_paths(dir):
    """
    Scansiona la cartella del dataset UC Merced e recupera i percorsi dei file .tif.
    La classe di appartenenza viene recuperata dal nome della sottocartella.
    """
    paths = []
    classi = []
    for r, d, files in os.walk(dir):
        for file in files: 
            if file.endswith(".tif"):
                full_path  = os.path.join(r, file)
                paths.append(full_path)
                classi.append(os.path.basename(r))
    return paths, classi

def crea_istogrammi(paths, kmeans_model, k):
    """
    Traduce ogni immagine in un istogramma.
    Per ogni immagine estrae i descrittori SIFT, li assegna alla Word 
    più vicina e conta le occorrenze.
    """
    sift = cv2.SIFT_create()
    istogrammi = []
    
    for path in paths:
        img = cv2.imread(path)
        if img is not None:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Estrazione descrittori
            _, descrittori = sift.detectAndCompute(img_gray, None) 
            
            if descrittori is not None:
                # Normalizzazione L2
                descrittori = normalize(descrittori, norm="l2")

                # Assegnazione di ogni descrittore al centroide più vicino
                words = kmeans_model.predict(descrittori)

                # Costruzione istogramma tramite frequenze Word
                hist, _ = np.histogram(words, bins=np.arange(k + 1))
            else:
                # Gestione di immagini anomale senza keypoints
                hist = np.zeros(k)
                
            istogrammi.append(hist)
            
    matrice_istogrammi = np.vstack(istogrammi)

    # Normalizzazione L1
    return normalize(matrice_istogrammi, norm='l1')

def valuta_classificatori(matrice_istogrammi, classi):
    """
    Valuta SVM e Random Forest tramite 3-Fold Cross-Validation Stratificata.
    """
   
    classi = np.array(classi)
    # La stratificazione garantisce che ogni fold abbia la stessa proporzione di classi
    s = StratifiedKFold(n_splits=3, shuffle = True, random_state= 42)



    voti_svm_acc = []
    voti_svm_prec = []
    voti_svm_rec = []
    voti_svm_f1 = []
    
    voti_rf_acc = []
    voti_rf_prec = []
    voti_rf_rec = []
    voti_rf_f1 = []

    for train, test in s.split(matrice_istogrammi, classi):
        X_train, y_train = matrice_istogrammi[train], classi[train]
        X_test, y_test = matrice_istogrammi[test], classi[test]

        # Reset dei modelli ad ogni iterazione
        svm = SVC(kernel="rbf", random_state=42)
        rf = RandomForestClassifier(random_state=42)

        # -- SVM --
        svm.fit(X_train, y_train)
        predizione_svm = svm.predict(X_test)
        report_svm = classification_report(y_test, predizione_svm, output_dict=True, zero_division=0)

        voti_svm_acc.append(report_svm['accuracy'])
        voti_svm_prec.append(report_svm['macro avg']['precision'])
        voti_svm_rec.append(report_svm['macro avg']['recall'])
        voti_svm_f1.append(report_svm['macro avg']['f1-score'])

        # -- Random Forest --
        rf.fit(X_train, y_train)
        predizione_rf = rf.predict(X_test)
        report_rf = classification_report(y_test, predizione_rf, output_dict= True, zero_division=0)

        voti_rf_acc.append(report_rf['accuracy'])
        voti_rf_prec.append(report_rf['macro avg']['precision'])
        voti_rf_rec.append(report_rf['macro avg']['recall'])
        voti_rf_f1.append(report_rf['macro avg']['f1-score'])

    # Media delle metriche sui 3 fold
    risultati_medi = {
            "SVM_RBF": {
                "acc": np.mean(voti_svm_acc),
                "prec": np.mean(voti_svm_prec),
                "rec": np.mean(voti_svm_rec),
                "f1": np.mean(voti_svm_f1)
            },
            "Random_Forest": {
                "acc": np.mean(voti_rf_acc),
                "prec": np.mean(voti_rf_prec),
                "rec": np.mean(voti_rf_rec),
                "f1": np.mean(voti_rf_f1)
            }
        }
    
    return risultati_medi

def seleziona_e_salva_vincitore(paths_uc, classi_uc, valori_k):
    """
    Cicla sui valori di K, esegue le valutazioni, decreta il modello con il miglior
    F1-Score, genera la relativa matrice di confusione e salva il modello addestrato.
    """
    miglior_f1 = 0
    miglior_k = None
    miglior_nome = None
    # Dizionario per salvare le matrici BoW
    matrici_salvate = {}
    
    for k in valori_k:
        print(f"Analisi con vocabolario K = {k}")

        nome_file_vocabolario = f"vocabolario_k{k}.pkl"
        
        try:
            with open(nome_file_vocabolario, "rb") as f:
                kmeans_caricato = pickle.load(f)
        except FileNotFoundError:
            print(f"File {nome_file_vocabolario} non trovato")
            continue
            
        print(f"Elaborazione feature per K = {k}")

        matrice_X = crea_istogrammi(paths_uc, kmeans_caricato, k)
        matrici_salvate[k] = matrice_X

        risultati = valuta_classificatori(matrice_X, classi_uc)
        
        for nome_modello, metriche in risultati.items():
            print(f"Modello: {nome_modello} | Acc: {metriche['acc']:.4f} | Prec (Macro): {metriche['prec']:.4f} | Rec (Macro): {metriche['rec']:.4f} | F1-Macro: {metriche['f1']:.4f}")
            
            if metriche['f1'] > miglior_f1:
                miglior_f1 = metriche['f1']
                miglior_k = k
                miglior_nome = nome_modello

    print(f"\nMiglior modello: {miglior_nome} con K = {miglior_k}\n")
    print(f" F1-Score: {miglior_f1:.4f}")

    # Matrice confusione

    X_definitiva = matrici_salvate[miglior_k]
    y_definitiva = np.array(classi_uc)
    
    if miglior_nome == "SVM_RBF":
        modello_finale = SVC(kernel="rbf", random_state=42)
    else:
        modello_finale = RandomForestClassifier(random_state=42)

    print("Generazione della Matrice di Confusione per il miglior modello")
    nomi_classi = np.unique(y_definitiva)
    predizioni_cv = cross_val_predict(modello_finale, X_definitiva, y_definitiva, cv=3)
    cm = confusion_matrix(y_definitiva, predizioni_cv, labels=nomi_classi)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomi_classi)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    
    plt.title(f"Matrice di Confusione {miglior_nome} (K={miglior_k})")
    plt.tight_layout()
    
    nome_img_matrice = "matrice_di_confusione.png"
    plt.savefig(nome_img_matrice, dpi=300)
    print(f"Matrice salvata: {nome_img_matrice}")
    
        
    # Addestramento finale sul 100% dei dati disponibili
    print("Addestramento finale su tutto il dataset")
    modello_finale.fit(X_definitiva, y_definitiva)

    nome_file_modello = "miglior_classificatore.pkl"
    with open(nome_file_modello, "wb") as f:
        pickle.dump(modello_finale, f)
        
    print(f"Modello salvato: {nome_file_modello}")
   


if __name__ == "__main__":
    
    cartella_ucmerced = "./UCMerced_LandUse/Images"
    
    
    print("Ricerca immagini nel dataset UC Merced")
    paths_uc, classi_uc = get_image_paths(cartella_ucmerced)
    print(f"Trovate {len(paths_uc)} immagini pronte")
    
    valori_k = [50, 100, 500] 

    seleziona_e_salva_vincitore(paths_uc, classi_uc, valori_k)