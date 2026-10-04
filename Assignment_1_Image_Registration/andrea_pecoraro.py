import cv2
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ====================================================================
# FUNZIONI DI CARICAMENTO E SALVATAGGIO
# ====================================================================

def riempi_lista(percorso):
    """
    Scansiona la directory specificata per caricare le coppie di immagini.
    Restituisce una lista di dizionari contenenti i nomi delle coppie e le immagini
    """
    lista = []
    for i in range(1, 9):
        nome_cartella = f"c{i}"
        path_cartella = os.path.join(percorso, nome_cartella)
        if os.path.isdir(path_cartella):
            r_path = None
            t_path = None
            for file in os.listdir(path_cartella):
                name, end = os.path.splitext(file)
                if name.endswith("R"):
                    r_path = os.path.join(path_cartella, file)
                elif name.endswith("T"):
                    t_path = os.path.join(path_cartella, file)

            if r_path is not None and t_path is not None:
                img_r = cv2.imread(r_path)
                img_t = cv2.imread(t_path)
                
                if img_r is None or img_t is None:
                     print(f"Errore lettura file nella cartella {nome_cartella}")
                else:                    
                    lista.append({"coppia" : nome_cartella, "percorso": percorso, "img_r": img_r, "img_t": img_t})
                
    return lista


def ottieni_img():
    """
    Gestisce l'input utente per il caricamento di immagini custom.
    """
    print("Scrivi il path dell'immagine reference: \n")
    path = input()
    if os.path.isfile(path):
        img_ref = cv2.imread(path)
    else: return (None, None)
    print("Scrivi il path dell'immagine moving: \n")
    path = input()
    if os.path.isfile(path):
        img_mov = cv2.imread(path)
    else: return (None, None)
    return (img_ref, img_mov)


def salva_immagini_test(img_r, img_t, tx_stimato, ty_stimato, theta_stimato, coppia):
    """Applica la trasformazione finale e salva le immagini allineate e di differenza."""
    h, w = img_r.shape[:2]
    c_fin = np.cos(theta_stimato)
    s_fin = np.sin(theta_stimato)
    
    M_finale = np.array([
        [ c_fin, -s_fin, tx_stimato],
        [s_fin, c_fin, ty_stimato]
    ], dtype=np.float32)
    
    # Applica la trasformazione all'immagine moving
    img_allineata = cv2.warpAffine(img_t, M_finale, (w, h))
    # Calcola l'immagine differenza 
    img_differenza = cv2.absdiff(img_r, img_allineata)
    
    # Salva le immagini
    cv2.imwrite(f"Test_Allineata_{coppia}.png", img_allineata)
    cv2.imwrite(f"Test_Differenza_{coppia}.png", img_differenza)



def salva_grafico_mi(storia_mi, metodo, b, coppia):
    """Crea e salva il grafico dell'andamento della Mutua Informazione."""
    if not storia_mi:  # Controllo di sicurezza
        return
        
    plt.figure()
    plt.plot(storia_mi)
    plt.title(f"Andamento MI - {metodo} - Bins: {b} ({coppia})")
    plt.xlabel("Valutazioni della funzione")
    plt.ylabel("Mutua Informazione")
    plt.grid(True)
    plt.savefig(f"Andamento_MI_{metodo}_{b}_{coppia}.png")
    plt.close()



# ====================================================================
# CALCOLO MUTUA INFORMAZIONE E TRASFORMAZIONE
# ====================================================================

def mutua_informazione(img_r, img_t, bins, maschera = None):
    """
    Calcola l'istogramma congiunto e la Mutua Informazione tra due immagini.
    """

    if maschera is not None:

        pixel_validi = maschera > 0
        array_r = img_r[pixel_validi]
        array_t = img_t[pixel_validi]
        
        if len(array_r) < 100: 
            return 0.0
    else:
        array_r = img_r.ravel()
        array_t = img_t.ravel()

    #calcolo istogramma con np.histogram2d(), poichè accetta un array a una dimensione, utilizziamo ravel() per convertire matrici 2D in 1D
    istogramma, _, _ = np.histogram2d(
        array_r, 
        array_t, 
        bins = bins, 
        range=[[0, 256], [0, 256]]
    )

    # Normalizzazione e calcolo probabilità marginali
    p_xy = istogramma / float(np.sum(istogramma))
    p_x = np.sum(p_xy, axis=1) # Somma sulle colonne
    p_y = np.sum(p_xy, axis=0) # Somma sulle righe

    # Calcolo delle Entropie 
    # Filtriamo i valori > 0 per evitare l'errore log(0)
    p_x_nz = p_x[p_x > 0]
    h_x = -np.sum(p_x_nz * np.log(p_x_nz))

    p_y_nz = p_y[p_y > 0]
    h_y = -np.sum(p_y_nz * np.log(p_y_nz))
    
    p_xy_nz = p_xy[p_xy > 0]
    h_xy = -np.sum(p_xy_nz * np.log(p_xy_nz))
    
    mi = h_x + h_y - h_xy
    return mi

def mi_tran(par, img_r, img_t, bins, storia_mi=None):
    """
    Applica le trasformazioni geometriche passate dall'ottimizzatore e restituisce 
    la MI valutata sul nuovo allineamento.
    """

    tx, ty, theta_deg = par 
    theta_rad = np.radians(theta_deg)

    # matrice di trasformazione
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)

    matrice = np.array([
        [c,  -s, tx],
        [s, c, ty]
    ], dtype=np.float32)
    
    h, w = img_r.shape[:2]

    # trasformazione con interpolazione bilineare
    img_t_tran = cv2.warpAffine(img_t, matrice, (w, h), flags = cv2.INTER_LINEAR)

    # Maschera per escludere i bordi neri dal calcolo della MI
    maschera_iniziale = np.ones((h, w), dtype=np.uint8) * 255 
    maschera_tran = cv2.warpAffine(maschera_iniziale, matrice, (w, h), flags=cv2.INTER_LINEAR)

    mi = mutua_informazione(img_r, img_t_tran, bins, maschera= maschera_tran)

    if storia_mi is not None:
        storia_mi.append(mi)

    return -mi

# ====================================================================
#  STIMA E OTTIMIZZAZIONE MODELLO
# ====================================================================

def stima_modello(lista, metodo, gt_df, nome_set, bins, salva_immagini=False, traccia_mi=False):
    """
    Applica pre-processing, lancia l'ottimizzatore 
    e valuta le metriche confrontandole con il Ground Truth.
    """
    if not isinstance(bins, list):
        bins = [bins]
    x_0 = [0.0, 0.0, 0.0]
    risultati_metodo = []

    for b in bins:
        result = []
        for elem in lista:
            img_r = elem["img_r"]
            img_t = elem["img_t"]
            coppia = elem["coppia"]

            # Pre-processing: convesione a scala di grigi e sfocatura con GaussianBlur
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            img_t_gray = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)

            img_r_blur = cv2.GaussianBlur(img_r_gray, (13, 13), 0)
            img_t_blur = cv2.GaussianBlur(img_t_gray, (13, 13), 0)

            storia_mi = [] if traccia_mi else None

            if metodo == "Nelder-Mead":
                # Simplesso iniziale per aiutare la convergenza
                simplesso_iniziale = np.array([
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0]
                ])
                risultato = minimize(mi_tran, x_0, args=(img_r_blur, img_t_blur, b, storia_mi), method=metodo,
                                     options={'initial_simplex': simplesso_iniziale})
            else :
                risultato = minimize(mi_tran, x_0, args=(img_r_blur, img_t_blur, b, storia_mi), method=metodo)

            if traccia_mi and (elem == lista[0] or elem == lista[1]):
                salva_grafico_mi(storia_mi, metodo, b, coppia)

            tx_stimato, ty_stimato, theta_deg_stimato = risultato.x
            theta_stimato = np.radians(theta_deg_stimato)

            tx_reale, ty_reale, theta_reale = 0.0, 0.0, 0.0

            # confronto Ground Truth
            for riga in gt_df:
                pair_gt = riga['Pair']
                dataset_gt = riga['Dataset']
                if pair_gt == coppia and dataset_gt == nome_set:
                    tx_reale = riga['Tx']
                    ty_reale = riga['Ty']
                    theta_reale = riga['AngleRad']
                    break

            # Calcolo Errori
            err_traslazione = np.linalg.norm([tx_stimato - tx_reale, ty_stimato - ty_reale])
            err_theta = abs(theta_stimato - theta_reale)

            result.append({
                'coppia': coppia,
                'err_theta': err_theta,
                'err_traslazione' : err_traslazione
            })

            if salva_immagini:
                salva_immagini_test(img_r, img_t, tx_stimato, ty_stimato, theta_stimato, coppia)

        errori_theta = [r['err_theta'] for r in result]
        errori_traslazione = [r['err_traslazione'] for r in result]

        media_theta = np.mean(errori_theta)
        media_traslazione = np.mean(errori_traslazione)

        ds_theta = np.std([r['err_theta'] for r in result])
        ds_traslazione = np.std([r['err_traslazione'] for r in result])

        print(f"Metodo: {metodo:12} | Bins: {b:3} | "
              f"Err Trasl: {media_traslazione:5.2f} ± {ds_traslazione:4.2f} px | "
              f"Err Ang: {media_theta:6.4f} ± {ds_theta:5.4f} rad")
        
        # Salviamo la statistica per la grid search
        risultati_metodo.append({
            'metodo': metodo,
            'bins': b,
            'err_traslazione': media_traslazione,
            'ds_traslazione' : ds_traslazione,
            'err_theta': media_theta,
            'ds_theta' : ds_theta
            
        })

    return risultati_metodo


def stima_modello_custom(img_ref, img_mov, metodo, bins):
    """
    Versione ridotta di 'stima_modello' dedicata esclusivamente all'applicazione su 
    immagini custom fornite dall'utente.
    """

    # Pre-processing: convesione a scala di grigi e sfocatura con GaussianBlur
    img_r_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    img_t_gray = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)

    img_r_blur = cv2.GaussianBlur(img_r_gray, (13, 13), 0)
    img_t_blur = cv2.GaussianBlur(img_t_gray, (13, 13), 0)

    storia_mi = [] 
    x_0 = [0.0, 0.0, 0.0]

    if metodo == "Nelder-Mead":
        # Simplesso iniziale per aiutare la convergenza
        simplesso_iniziale = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        risultato = minimize(mi_tran, x_0, args=(img_r_blur, img_t_blur, bins, storia_mi), method=metodo,
                                options={'initial_simplex': simplesso_iniziale})
    else :
        risultato = minimize(mi_tran, x_0, args=(img_r_blur, img_t_blur, bins, storia_mi), method=metodo)

    tx_stimato, ty_stimato, theta_deg_stimato = risultato.x
    theta_stimato = np.radians(theta_deg_stimato)

    # Salvataggio Grafico e Immagini passando la stringa "Custom" come nome della coppia
    salva_grafico_mi(storia_mi, metodo, bins, "Custom")
    salva_immagini_test(img_ref, img_mov, tx_stimato, ty_stimato, theta_stimato, "Custom")
    print(f"Traslazione: (Tx: {tx_stimato:.2f} px, Ty: {ty_stimato:.2f} px)")
    print(f"Rotazione: {theta_deg_stimato:.4f} gradi")


def esegui_grid_search(lista, gt_df, nome_set, bins):
    """Confronta diversi metodi e configurazioni per trovare la migliore."""
    tutti = []
    for m in ["Powell", "Nelder-Mead", "BFGS"]:
        tutti.extend(stima_modello(lista, m, gt_df, nome_set, bins))
    
    best = min(tutti, key=lambda x: x['err_traslazione'])
    print(f"\nMetodo Migliore: {best['metodo']} con {best['bins']} bins "
          f"(Err Traslazione: {best['err_traslazione']:.2f} px, "
          f"Err Theta: {best['err_theta']:.4f} rad)")
    return best


def test_nuova_coppia(metodo, bins):
    """caricamento e l'allineamento di nuove coppie."""
    img_r, img_t = ottieni_img()
    if img_r is None or img_t is None:
        print("errore nel caricamento delle immagini")
        return 
    stima_modello_custom(img_r, img_t, metodo, bins)
            
# ====================================================================
# MAIN 
# ====================================================================

if __name__ == "__main__":
    print("Python sta cercando i file qui:", os.getcwd())
    gt_df = np.genfromtxt("GT.csv", delimiter=";", dtype=None, names=True)

    lista_val = riempi_lista("val")
    lista_test = riempi_lista("test")

    bins = [64, 128, 256]
    # FASE DI VALIDAZIONE 
    print("--- Ricerca iper-parametri sul Validation Set ---")
    best = esegui_grid_search(lista_val, gt_df, "val", bins)

    # GENERAZIONE GRAFICI VAL
    print(f"\n--- Generazione Grafici per  ({best['metodo']}, {best['bins']} bins) ---")
    # Usiamo lista_val[:2] per tracciare la MI solo per i primi due esperimenti
    stima_modello(lista_val[:2], best['metodo'], gt_df, "val", best['bins'], traccia_mi=True)

    # FASE DI TEST 
    print(f"\n--- Valutazione finale sul Test Set usando {best['metodo']} ---")

    risultati_test = stima_modello(
        lista_test, 
        best['metodo'], 
        gt_df, 
        "test", 
        best['bins'], 
        salva_immagini=True, 
        
    )
    # CUSTOM TEST
    scelta = input("Vuoi effettuare una test con una nuova coppia? (si, no)\n")
    if scelta == "si":
        test_nuova_coppia(best['metodo'], best['bins'])







        




            
    
                


        

