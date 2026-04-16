# DeepScene 360: Neural Radiance Fields (NeRF) for Multi-Object Scene Synthesis

## 1. Definirea Problemei și a Obiectivului
Proiectul vizează implementarea și optimizarea unei rețele neuronale de tip **Neural Radiance Fields (NeRF)** pentru reconstrucția unei scene volumetrice complexe din imagini 2D. Obiectivul central este **Sinteza de Vederi Noi (Novel View Synthesis)** — generarea de imagini fotorealiste din unghiuri neprevăzute în setul de antrenare, demonstrând capacitatea modelului de a învăța geometria implicită și ocluziile.
## 2. Descriere Dataset
Se utilizează setul de date academic **"NeRF Synthetic Dataset" (Blender)**, cu accent pe scena **Lego**.
* **Rezoluție**: Imaginile originale de 800x800 au fost scalate la **400x400 pixeli** pentru a reduce consumul de memorie video (VRAM) cu 75%.
* **Metadate**: Include parametrii intrinseci și matricile de rotație/translație (Camera-to-World) extrase din fișierele `.json`.
* **Protocol**: Split riguros Train / Validation / Test pentru monitorizarea precisă a performanței.

## 3. Abordare Propusă (Arhitectură și Matematică)
Modelul este un **Multi-Layer Perceptron (MLP)** profund care aproximează o funcție continuă 5D: $F_{\Theta}:(x,d)\rightarrow(c,\sigma)$.

### Decizii Tehnice și Hiperparametri:
* **Positional Encoding**: Proiectarea coordonatelor în frecvențe înalte pentru a captura detaliile fine ale texturii.
* **Eșantionare Ierarhică**: Utilizarea a $N_c=64$ eșantioane *Coarse* și $N_f=128$ eșantioane *Fine* pentru a optimiza randarea volumelor goale.
* **Optimizator**: Adam cu rată de învățare inițială $5\times10^{-4}$ și decădere exponențială.
* **Center Cropping**: Strategie aplicată în primele 500 de iterații pentru a forța convergența pe obiectul central.

## 4. Rezultate și Performanță (Benchmark)
Performanța a fost evaluată pe parcursul a **150.000 de iterații**, obținând un nivel de fotorealism în care artefactele sunt imperceptibile:

| Iterație (k) | PSNR (dB) | SSIM |
| :--- | :--- | :--- |
| 10 | 26.47 | 0.9023 |
| 50 | 30.36 | 0.9545 |
| 100 | 31.39 | 0.9652 |
| **150 (Final)** | **32.02** | **0.9695** |

**Compararea cu Baseline-ul Academic**: Implementarea curentă a reținut peste **98%** din performanța modelului original (32.54 dB), reprezentând un compromis eficient între resurse și calitate.

## 5. Probleme Întâmpinate și Soluții
* **Instabilitatea Mediului (Timeouts)**: S-a implementat un mecanism de **Micro-Checkpointing** la fiecare 10.000 de iterații, salvând starea optimizatorului pentru a permite reluarea antrenării în caz de deconectare.
* **Anomalii în Hărțile de Adâncime**: Erorile de tip "NaN" au fost eliminate prin neutralizare matematică (`nan_to_num`) și limitare statistică (clipping) între percentilele 1% și 99%.

## 6. Structură Repository
Repository-ul este organizat pentru a suporta multiple configurații de scene:
```
├── configs/            # Fișiere de configurare (lego.txt, chair.txt, etc.)
├── imgs/               # Resurse vizuale și diagrame de pipeline
├── load_*.py           # Scripturi dedicate pentru diferite tipuri de date (Blender, LLFF)
├── run_nerf.py         # Scriptul principal de training și inferență
├── run_nerf_helpers.py # Funcții auxiliare, arhitectura MLP și encodări
├── requirements.txt    # Dependențe (PyTorch, NumPy, OpenCV)
└── README.md           # Documentația tehnică curentă
```

## 7. Instalare și Utilizare
Proiectul necesită un mediu Python 3.8+ și suport CUDA pentru performanță optimă.

1.  **Pregătirea Mediului**:
    ```bash
    # Instalarea dependențelor necesare
    pip install -r requirements.txt
    ```

2.  **Antrenarea Modelului**:
    Pentru a porni procesul de optimizare folosind configurația specifică scenei Lego:
    ```bash
    python run_nerf.py --config configs/lego.txt
    ```
    *Notă: Scriptul utilizează automat `run_nerf_helpers.py` pentru arhitectura MLP și `load_blender.py` pentru încărcarea datelor.*

3.  **Randare și Inferență**:
    Odată ce modelul este antrenat (checkpoint-urile sunt salvate în folderul de log-uri), se poate genera un video fly-by 360°:
    ```bash
    python run_nerf.py --config configs/lego.txt --render_only --render_test
    ```

## 8. Concluzii și Direcții Viitoare
Implementarea a demonstrat succesul rețelelor de radianță neurală în capturarea geometriei complexe și a proprietăților materialelor (reflexii pe piesele Lego). Cu un PSNR final de **32.02 dB**, modelul produce rezultate aproape identice cu fotografiile reale de referință.

**Direcții de dezvoltare:**
* **Optimizarea Vitezei**: Integrarea unor structuri de date accelerate precum *Octrees* sau *Hash Grids* (inspirat de Instant-NGP) pentru a reduce timpul de antrenare de la ore la minute.
* **Generalizare**: Trecerea de la un model dependent de o singură scenă la o arhitectură care poate procesa scene noi cu un număr minim de imagini (Few-shot learning), prin adăugarea unor straturi de extracție a trăsăturilor (Feature Extractors).
* **Compresie**: Explorarea tehnicilor de pruning pentru a reduce dimensiunea checkpoint-urilor salvate, facilitând rularea pe dispozitive mobile.

---
*Proiect realizat de Dan Oprean (Universitatea "1 Decembrie 1918" din Alba Iulia), Aprilie 2026.*
