import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

PATH_UCM = "./UCMerced_LandUse/Images"

# ==========================================
# GESTIONE DATI
# ==========================================
def get_data():
    """Carica i dataset Validation e Test precedentemente preparati durante il training"""
    if not (os.path.exists('X_val_ucm.npy') and os.path.exists('X_test_ucm.npy')):
        print("File di split non trovati")
        return None, None, None, None

    print("Caricamento split di validazione e test")
    X_val = np.load('X_val_ucm.npy')
    y_val = np.load('y_val_ucm.npy')
    X_test = np.load('X_test_ucm.npy')
    y_test = np.load('y_test_ucm.npy')
    
    return X_val, y_val, X_test, y_test

# ==========================================
# METRICHE E VALUTAZIONE
# ==========================================
def get_f1_score(model, X, y_true_encoded):
    """Calcola l'F1-Score"""
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    y_true = np.argmax(y_true_encoded, axis=1)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report['macro avg']['f1-score']

def test_final_model(model, model_name, X_test, y_test):
    """
    Esegue il modello vincente sul test set
    Stampa le metriche e genera la matrice di confusione.
    """
    print(f"\nValutazione modello: {model_name}\n")
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Stampa Precision, Recall, F1 per ogni classe
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Recupera i nomi delle classi leggendo la cartella
    if os.path.exists(PATH_UCM):
        classi = sorted([d for d in os.listdir(PATH_UCM) if os.path.isdir(os.path.join(PATH_UCM, d))])
    else:
        classi = [str(i) for i in range(21)]
    # Costruisce e salva la matrice di confusione
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classi)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    
    img_name = f"cm_{model_name}.png"
    plt.savefig(img_name, dpi=300)
    print(f"Matrice di confusione salvata come: {img_name}\n")

# ==========================================
# LOGICA DI CONTROLLO PRINCIPALE
# ==========================================
def pipeline_valutazione(X_val, y_val, X_test, y_test):
    """Gestisce la pipeline di selezione e valutazione dei modelli"""
    print("Caricamento modelli in memoria")
    try:
        model_m1 = load_model('miglior_modello_metodo_1.keras')
        model_m2 = load_model('miglior_modello_metodo_2.keras')
    except Exception as e:
        print(f"Caricamento fallito: {e}")
        return
    
    print("Parametri Metodo 1:")
    model_m1.summary()
    
    print("Parametri Metodo 2:")
    model_m2.summary()

    print("\nSelezione Modello")
    f1_m1 = get_f1_score(model_m1, X_val, y_val)
    f1_m2 = get_f1_score(model_m2, X_val, y_val)
    
    print(f"Metodo 1 (TL) F1-Score:      {f1_m1:.4f}")
    print(f"Metodo 2 (Scratch) F1-Score: {f1_m2:.4f}")
    
    print("\nRisultato Selezione")
    if f1_m1 >= f1_m2:
        print("Modello selezionato: Metodo 1")
        test_final_model(model_m1, "Metodo_1", X_test, y_test)
    else:
        print("Modello selezionato: Metodo 2")
        test_final_model(model_m2, "Metodo_2", X_test, y_test)


if __name__ == "__main__":
    print("Inizio valutazione")
    X_val, y_val, X_test, y_test = get_data()
    
    if X_val is not None:
        pipeline_valutazione(X_val, y_val, X_test, y_test)


    