import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

LARGHEZZA_IMG = 128
ALTEZZA_IMG = 128
# Medie del dataset ImageNet in formato BGR (usate per lo zero-centering)
MEDIA_IMAGENET_BGR = np.array([103.939, 116.779, 123.68]) 

# ==========================================
# ELABORAZIONE DATASET
# ==========================================
def load_and_preprocess_dataset(base_path):
    """
    Carica le immagini tramite OpenCV, esegue il resize a 128x128 
    e applica lo zero-centering basato sulle medie di ImageNet
    """
    immagini = []
    labels = []
    # Estrae e ordina alfabeticamente i nomi delle cartelle
    classi = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    # Crea un dizionario per mappare ogni nome di classe a un ID numerico
    id_classi = {nome_classe: i for i, nome_classe in enumerate(classi)}
    
    for nome_classe, id_classe in id_classi.items():
        path_classe = os.path.join(base_path, nome_classe)
        
        for nome_file in os.listdir(path_classe):
            path_img = os.path.join(path_classe, nome_file)
            img = cv2.imread(path_img)
            
            if img is not None:
                # Resize per avere input uniformi nella rete
                img = cv2.resize(img, (LARGHEZZA_IMG, ALTEZZA_IMG))
                img = img.astype('float32')
                # Zero-centering
                img -= MEDIA_IMAGENET_BGR 
                
                immagini.append(img)
                labels.append(id_classe)
                
    X = np.array(immagini)
    y = np.array(labels)
    # Converte le label numeriche in vettori One-Hot
    y_cod = to_categorical(y, num_classes=len(id_classi))
    
    return X, y_cod, id_classi

# ==========================================
# ARCHITETTURA DEL MODELLO
# ==========================================
def custom_residual_block(x, filtri, dim_kernel=3):
    """Implementa un blocco residuale custom con skip connection"""

    # Salva l'input originale per sommarlo alla fine
    skip_connection = x
    
    # Primo layer convoluzionale del blocco
    x = Conv2D(filtri, dim_kernel, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Secondo layer convoluzionale del blocco
    x = Conv2D(filtri, dim_kernel, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    
    # Controllo dimensionale: se il numero di filtri è cambiato rispetto all'input,
    # applica una Conv2D 1x1 alla skip connection per allineare i tensori prima della somma.
    if skip_connection.shape[-1] != filtri:
        skip_connection = Conv2D(filtri, (1, 1), padding='same', activation=None)(skip_connection)
        skip_connection = BatchNormalization()(skip_connection)
    
    # Somma l'input processato con l'input originale
    x = Add()([skip_connection, x])
    x = Activation('relu')(x)
    
    return x

def build_custom_resnet(input_shape=(128, 128, 3), num_classes=30):
    """Costruisce l'architettura ResNet custom per la classificazione"""
    inputs = Input(shape=input_shape)
    
    # Feature extraction iniziale
    x = Conv2D(32, (3, 3), padding='same', activation=None)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Blocchi residuali
    x = custom_residual_block(x, filtri=64)
    x = MaxPooling2D((2, 2))(x) 
    
    x = custom_residual_block(x, filtri=128)
    x = MaxPooling2D((2, 2))(x)
    
    x = custom_residual_block(x, filtri=256)
    x = MaxPooling2D((2, 2))(x)
    
    # Classificatore MLP 
    x = Flatten()(x) 
    
    # Primo layer Denso 
    x = Dense(512, activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x) 

    # Secondo layer Denso intermedio
    x = Dense(256, activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    # Layer di Output
    outputs = Dense(num_classes, activation='softmax', name='output_classificatore')(x)
    
    modello = Model(inputs=inputs, outputs=outputs)
    return modello