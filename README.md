# Computer Vision & Pattern Recognition Portfolio

Questo repository contiene una suite di tre progetti pratici sviluppati per il corso di Visione Artificiale all'Università degli Studi di Palermo. La suite affronta l'intera evoluzione delle tecniche di analisi di immagini aeree e satellitari, dall'elaborazione geometrica classica fino al Deep Learning.

## 📌 Progetti Inclusi

### 1. Registrazione di Immagini Multimodali (RGB/IR)
Algoritmo per l'allineamento geometrico di coppie di immagini eterogenee (sensori RGB e termici).
* **Tecniche:** Massimizzazione della Mutua Informazione, algoritmi di ottimizzazione diretta (Powell, Nelder-Mead, BFGS).
* **Codice e Presentazione:** [Vai alla cartella](./Assignment_1_Image_Registration)

### 2. Classificazione Bag-of-Visual-Words
Pipeline di Machine Learning per la classificazione multi-classe di scenari urbani.
* **Tecniche:** Estrazione descrittori locali (SIFT), generazione vocabolari visivi (MiniBatchKMeans), addestramento e cross-validation di classificatori SVM (Kernel RBF) e Random Forest.
* **Codice e Presentazione:** [Vai alla cartella](./Assignment_2_Bag_of_Visual_Words)

### 3. Classificazione Deep Learning (Custom ResNet)
Progettazione e sviluppo *from scratch* di una Rete Neurale Convoluzionale residuale per la classificazione di immagini satellitari.
* **Tecniche:** Blocchi residuali custom, Transfer Learning, Fine-tuning, Data Augmentation dinamica, mitigazione dell'overfitting.
* **Codice e Presentazione:** [Vai alla cartella](./Assignment_3_Custom_ResNet)

## 🛠 Stack Tecnologico
* **Linguaggio:** Python
* **Librerie e Framework:** TensorFlow / Keras, OpenCV, Scikit-Learn, SciPy, NumPy, Matplotlib.