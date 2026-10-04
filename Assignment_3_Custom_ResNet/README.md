# Assignment 3: Classificazione Deep Learning (Custom ResNet)

Progettazione e sviluppo *from scratch* di una Rete Neurale Convoluzionale per la classificazione di immagini aeree in 21 scenari territoriali (dataset UC Merced).

## 🎯 Obiettivo
Analizzare l'efficacia delle architetture residuali personalizzate ed eseguire uno studio prestazionale per confrontare l'addestramento da zero sul dominio target rispetto all'acquisizione di feature generaliste tramite Transfer Learning.

## ⚙️ Architettura e Implementazione
*   **Custom Residual Block:** Costruzione matematica di blocchi residuali con *skip connections* (identity mapping) e tensori di allineamento (layer Conv2D 1x1) per la corretta propagazione dei gradienti nei salti dimensionali.
*   **Architettura Globale:** Rete profonda da circa 9.7 milioni di parametri, composta da layer convoluzionali in cascata, layer di flattening e un classificatore Multi-Layer Perceptron (MLP).
*   **Regolarizzazione e Pre-processing:** Mitigazione dell'overfitting applicando Zero-Centering (sottrazione della media ImageNet BGR), Data Augmentation dinamica (ImageDataGenerator), Batch Normalization e Dropout al 50% nei layer densi.

## 🧠 Strategie di Addestramento a Confronto
1.  **Transfer Learning e Fine-Tuning:** Pre-addestramento supervisionato della rete su un dataset massivo (AID a 30 classi). Successivo congelamento del *backbone convoluzionale* e addestramento del solo classificatore finale sulle 21 classi di UC Merced (learning rate conservativo a 0.0001 con AdamW).
2.  **Training from Scratch:** Inizializzazione casuale dei pesi e addestramento diretto dell'intera rete esclusivamente sul dataset UC Merced.

## 📊 Risultati
Il modello addestrato *from scratch* ha mostrato prestazioni finali marginalmente superiori sul Validation Set (F1-Score: 0.8663) rispetto all'approccio con Transfer Learning (0.8607). Valutato sul Test Set indipendente, ha ottenuto una **Accuracy globale dell'84%** e una precisione del 100% nel riconoscimento di classi complesse come aeroplani e campi da tennis.

**Tecnologie:** Python, TensorFlow / Keras, OpenCV, Scikit-Learn.