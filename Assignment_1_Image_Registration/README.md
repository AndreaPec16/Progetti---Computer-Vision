# Assignment 1: Registrazione di Immagini Multimodali (RGB/IR)

Questo progetto affronta il problema dell'allineamento geometrico di coppie di immagini aeree eterogenee, acquisite rispettivamente da sensori ottici (RGB) e termici (Infrarosso).

## 🎯 Obiettivo
Stimare i parametri di una trasformazione di roto-traslazione (tx, ty, θ) per sovrapporre perfettamente l'immagine *moving* (Infrarosso) all'immagine *reference* (RGB). L'allineamento è guidato dalla massimizzazione della **Mutua Informazione (MI)** tra le due immagini.

## ⚙️ Dettagli Implementativi
*   **Calcolo Mutua Informazione:** Implementazione matematica per la generazione dell'istogramma congiunto 2D, estrazione delle distribuzioni di probabilità marginali e calcolo dell'entropia e dell'entropia congiunta.
*   **Gestione Maschere Dinamiche:** Implementazione di maschere logiche durante l'interpolazione bilineare (warp affine) per escludere i pixel neri di bordo dal calcolo della Mutua Informazione, prevenendo falsi positivi.
*   **Ottimizzazione:** Studio comparativo tramite SciPy per individuare il miglior algoritmo di ottimizzazione diretta, confrontando **Powell, Nelder-Mead e BFGS**.
*   **Grid Search degli Iper-parametri:** Ricerca della configurazione ottimale sul Validation Set variando la risoluzione dei bin dell'istogramma (64, 128, 256).

## 📊 Risultati
La valutazione sul Validation Set ha evidenziato che la combinazione ottimale è l'algoritmo **Nelder-Mead con 64 bin**. Tale configurazione, applicata al Test Set, ha permesso di allineare correttamente le immagini con un errore traslazionale medio di ~30.51 px e un errore angolare di ~0.105 rad.

**Tecnologie:** Python, OpenCV, SciPy, NumPy, Matplotlib.