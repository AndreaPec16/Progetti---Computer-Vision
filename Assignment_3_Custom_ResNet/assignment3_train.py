import os
# disabilita warning TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from assignment3_core import load_and_preprocess_dataset, build_custom_resnet

PATH_AID = "./AID"
PATH_UCM = "./UCMerced_LandUse/Images"

# ==========================================
# UTILITÀ PER GRAFICI
# ==========================================
def plot_history(history, titolo_modello):
    """Genera e salva i grafici di loss e accuracy"""
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{titolo_modello} - Accuracy')
    plt.xlabel('Epoca')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{titolo_modello} - Loss')
    plt.xlabel('Epoca')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{titolo_modello}_curve.png', dpi=300)
    plt.close()
    print(f"Curve salvate: {titolo_modello}_curve.png")

# ==========================================
# PREPARAZIONE DATI
# ==========================================

def get_dataset(file_x, file_y, source_path):
    """
    Carica il dataset da file numpy se esistono, altrimenti legge le immagini tramite OpenCV e salva i file
    """
    if os.path.exists(file_x) and os.path.exists(file_y):
        return np.load(file_x), np.load(file_y)
    
    X, y, _ = load_and_preprocess_dataset(source_path)
    np.save(file_x, X)
    np.save(file_y, y)
    return X, y

def prepare_datasets():
    """Crea split stratificati: AID (70/30) e UC Merced (70/15/15)."""
    # AID: 70% Train, 30% Val
    X_aid, y_aid = get_dataset('X_aid.npy', 'y_aid.npy', PATH_AID)
    X_tr_aid, X_v_aid, y_tr_aid, y_v_aid = train_test_split(
        X_aid, y_aid, test_size=0.30, stratify=y_aid, random_state=42
    )
    
    # Caricamento UC Merced
    X_ucm, y_ucm = get_dataset('X_ucm.npy', 'y_ucm.npy', PATH_UCM)
    
    # Se i file degli split esistono già, evitiamo di ricalcolarli
    if os.path.exists('X_val_ucm.npy') and os.path.exists('X_test_ucm.npy'):
        print("Split UC Merced presenti in memoria")
        X_tr_ucm, y_tr_ucm = np.load('X_train_ucm.npy'), np.load('y_train_ucm.npy')
        X_v_ucm, y_v_ucm = np.load('X_val_ucm.npy'), np.load('y_val_ucm.npy')
        X_ts_ucm, y_ts_ucm = np.load('X_test_ucm.npy'), np.load('y_test_ucm.npy')
    else:
        print("Generazione e salvataggio split UC Merced")
        # Primo split: separa il 70% di Training dal 30% 
        X_tr_ucm, X_tmp, y_tr_ucm, y_tmp = train_test_split(
            X_ucm, y_ucm, test_size=0.30, stratify=y_ucm, random_state=42
        )
        # Secondo split: divide il 30% in due
        X_v_ucm, X_ts_ucm, y_v_ucm, y_ts_ucm = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
        )
        # Salviamo gli split per il file di valutazione
        np.save('X_train_ucm.npy', X_tr_ucm); np.save('y_train_ucm.npy', y_tr_ucm)
        np.save('X_val_ucm.npy', X_v_ucm); np.save('y_val_ucm.npy', y_v_ucm)
        np.save('X_test_ucm.npy', X_ts_ucm); np.save('y_test_ucm.npy', y_ts_ucm)
    
    data_aid = {'train': (X_tr_aid, y_tr_aid), 'val': (X_v_aid, y_v_aid)}
    data_ucm = {'train': (X_tr_ucm, y_tr_ucm), 'val': (X_v_ucm, y_v_ucm), 'test': (X_ts_ucm, y_ts_ucm)}
    
    return data_aid, data_ucm

def get_augmenter():
    """data augmentation dinamica"""
    return ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20)

# ==========================================
# STRATEGIE DI TRAINING
# ==========================================
def pretrain_aid(data_aid):
    """Metodo 1 Pre-addestramento da zero su 30 classi AID"""
    print("\nMetodo 1: Pre-training su dataset AID")
    model = build_custom_resnet(num_classes=30)

    # AdamW usato per gestire la regolarizzazione dei pesi
    model.compile(optimizer=AdamW(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Early Stopping ferma il training appena val_loss sale
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    X_train, y_train = data_aid['train']
    X_val, y_val = data_aid['val']
    aug = get_augmenter()
    
    # Salviamo history per i grafici
    history = model.fit(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=50, callbacks=[early_stop])
    model.save('modello_preaddestrato_aid.keras')
    plot_history(history, "Curve_Preaddestramento_AID")
    
    return model

def finetune_ucm(pretrained_model, data_ucm):
    """Metodo 1 Fine-tuning su 21 classi UC Merced"""
    print("\nMetodo 1: Fine-tuning su dataset UC Merced")

    # Sostituzione del classificatore per le 21 classi di UC Merced
    base_output = pretrained_model.get_layer('output_classificatore').input
    new_output = Dense(21, activation='softmax', name='classifier_ucm')(base_output)
    model = Model(inputs=pretrained_model.input, outputs=new_output)

    # Congelamento dei layer convoluzionali e BatchNormalization
    for layer in model.layers:
        if 'conv2d' in layer.name or 'batch_normalization' in layer.name: 
            layer.trainable = False

    # Learning rate ridotto per non alterare drasticamente i pesi pre-addestrati
    model.compile(optimizer=AdamW(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    X_train, y_train = data_ucm['train']
    X_val, y_val = data_ucm['val']
    aug = get_augmenter()
    
    history = model.fit(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=50, callbacks=[early_stop])
    model.save('miglior_modello_metodo_1.keras')
    plot_history(history, "Curve_Metodo_1")
    
    return model

def train_solo_ucm(data_ucm):
    """Metodo 2 Addestramento da zero direttamente su 21 classi UC Merced."""
    print("\nMetodo 2: Addestramento da zero su dataset UC Merced")
    model = build_custom_resnet(num_classes=21)
    
    model.compile(optimizer=AdamW(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    X_train, y_train = data_ucm['train']
    X_val, y_val = data_ucm['val']
    aug = get_augmenter()
    
    history = model.fit(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=50, callbacks=[early_stop])
    model.save('miglior_modello_metodo_2.keras')
    plot_history(history, "Curve_Metodo_2")

    return model

if __name__ == "__main__":
    print("Inizializzazione pipeline")
    data_aid, data_ucm = prepare_datasets()

    model_aid = pretrain_aid(data_aid)
    best_model_s1 = finetune_ucm(model_aid, data_ucm)

    best_model_s2 = train_solo_ucm(data_ucm)
    
    print("\nPipeline completata")

