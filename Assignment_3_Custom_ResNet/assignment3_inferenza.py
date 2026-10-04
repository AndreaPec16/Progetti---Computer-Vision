import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from assignment3_core import LARGHEZZA_IMG, ALTEZZA_IMG, MEDIA_IMAGENET_BGR
# Lista fissa delle classi, evita di dover ricaricare l'intero dataset per mappare le classi.

classi_UCM = [
    'agricultural', 'airplane', 'baseball_diamond', 'beach', 'buildings',
    'chaparral', 'dense_residential', 'forest', 'freeway', 'golf_course',
    'harbor', 'intersection', 'medium_residential', 'mobile_homepark',
    'overpass', 'parking_lot', 'river', 'runway', 'sparse_residential',
    'storage_tanks', 'tennis_court'
]

# ==========================================
# PRE-PROCESSING 
# ==========================================
def preprocess_single_image(img_path):
    """Esegue caricamento e pre-processing di una singola immagine"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Errore caricamento immagine: {img_path}")
        return None

    img = cv2.resize(img, (LARGHEZZA_IMG, ALTEZZA_IMG))
    # Cast obbligatorio per sottrarre media ImageNet
    img = img.astype('float32')
    img -= MEDIA_IMAGENET_BGR 
    
    # La rete Keras si aspetta input sotto forma di batch (Es: shape [32, 128, 128, 3]).
    # expand_dims simula un batch di dimensione 1, portando la shape a [1, 128, 128, 3].
    return np.expand_dims(img, axis=0)

# ==========================================
# ESECUZIONE
# ==========================================
def run_inference(img_path, model_path):
    """
    Elabora l'immagine e restituisce le 5 classi più probabili 
    """
    img_tensor = preprocess_single_image(img_path)
    if img_tensor is None:
        return

    try:
        modello = load_model(model_path)
    except Exception as e:
        print(f"Caricamento modello fallito: {e}")
        return
    # Estrae le probabilità dalla softmax
    probs = modello.predict(img_tensor, verbose=0)[0]

    # argsort ordina in ordine crescente, [-5:] prende gli ultimi 5 e [::-1] inverte l'array mettendoli in ordine decrescente di probabilità.
    indici_top_5 = np.argsort(probs)[-5:][::-1]

    print(f"\nFile: {os.path.basename(img_path)}")
    print("Top 5 Predizioni:")
    for i, idx in enumerate(indici_top_5, start=1):
        print(f"  {i}. {classi_UCM[idx]} {probs[idx]*100:.1f}%")
    print("\n")


if __name__ == "__main__":
    best_model = 'miglior_modello_metodo_2.keras'
    
    target_img = input("Path immagine da classificare: ")
    
    if os.path.exists(target_img):
        run_inference(target_img, best_model)
    else:
        print(f"Errore, file {target_img} non trovato")