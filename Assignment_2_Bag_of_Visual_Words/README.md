# Assignment 2: Classificazione Bag-of-Visual-Words

Questo modulo implementa una pipeline di Machine Learning classico per la classificazione multi-classe di scenari territoriali (21 classi del dataset UC Merced), sfruttando l'estrazione di feature locali.

## 🎯 Obiettivo
Classificare immagini satellitari senza l'ausilio di Reti Neurali, codificando la geometria e le texture dell'immagine tramite il paradigma **Bag-of-Visual-Words (BoW)** e addestrando classificatori supervisionati *shallow*.

## ⚙️ Dettagli Implementativi
*   **Estrazione Feature e Vocabolario (Apprendimento Non Supervisionato):** Estrazione dei descrittori locali **SIFT** da un dominio generale (dataset AID). Per evitare il sovraccarico in RAM, è stato applicato un campionamento casuale bilanciato (max 100.000 descrittori).
*   **Clustering ad Alta Efficienza:** Raggruppamento dei descrittori in *Visual Words* tramite l'algoritmo **MiniBatchKMeans** con normalizzazione L2, valutando vocabolari di dimensione K = 50, 100, 500.
*   **Modellazione delle Immagini:** Assegnazione dei descrittori SIFT del dataset target (UC Merced) ai centroidi del vocabolario, generando istogrammi di frequenza normalizzati in norma L1.
*   **Classificazione (Apprendimento Supervisionato):** Addestramento e valutazione comparativa tra **Support Vector Machine (Kernel RBF)** e **Random Forest** validati tramite *3-Fold Stratified Cross-Validation*.

## 📊 Risultati
Lo studio ha decretato vincitore il modello **SVM con Kernel RBF e un vocabolario di K=500 visual words**, che ha raggiunto un F1-Score macro-average di **0.65**. Il modello si è dimostrato altamente efficace su texture naturali (foreste, chaparral), mostrando lievi criticità fisiologiche nella distinzione di densità residenziali strutturalmente identiche.

**Tecnologie:** Python, OpenCV, Scikit-Learn, NumPy, Pickle.