# DeepScene 360: Neural Radiance Fields (NeRF) for Multi-Object Scene Synthesis

## 1. Definirea Problemei si a Obiectivului
Proiectul vizeaza implementarea unei retele neuronale de tip Neural Radiance Fields (NeRF) capabila sa reconstruiasca o scena volumetrica complexa plecand de la un set limitat de imagini 2D. Obiectivul principal este "New View Synthesis" — generarea unor imagini fotorealiste din unghiuri de camera care nu au fost prezente in setul de antrenare, demonstrand capacitatea retelei de a invata geometria 3D si ocluziile.

## 2. Descriere Dataset
Se va utiliza setul de date "NeRF Synthetic Dataset" (Blender), axat pe scena "LEGO".
* **Structura**: Dataset-ul include imagini RGB si fisiere de metadate (transforms.json) care contin parametrii camerelor.
* **Protocol Evaluare**: Split riguros de tip Train / Validation / Test.
* **Input**: Imagini 800x800 insotite de matricile de rotatie si translatie ale camerei.

## 3. Abordare Propusa (Arhitectura)
Modelul implementat este un Multi-Layer Perceptron (MLP) profund, optimizat pentru randare volumetrica.



**Componente cheie:**
* **Positional Encoding**: Transformarea coordonatelor spatiale (x, y, z) in frecvente inalte pentru a captura detalii fine.
* **Volume Rendering**: Utilizarea integralei de randare pentru a acumula culorile si densitatile de-a lungul razelor de lumina.
* **Live Training Monitor**: Sistem de randare periodica pentru a observa vizual progresul retelei in timp real.

## 4. Calendarul Evaluarilor (Plan de Lucru)
Activitatea respecta calendarul stabilit in regulamentul disciplinei:

| Etapa | Saptamana | Continut Minim | Pondere |
| :--- | :--- | :--- | :--- |
| **Prezentarea 1** | **S4 - S5** | Definire problema, dataset, model, structura repo | **20%** |
| **Prezentarea 2** | **S8 - S9** | Pipeline implementat, experiment complet, metrici | **20%** |
| **Prezentarea Finala** | **S12 - S14** | Demo functional, rezultate finale, documentatie completa | **60%** |

## 5. Structura Initiala Repository
Proiectul este organizat modular pentru a asigura claritatea si rulabilitatea codului:

```text
├── data/               # Imagini si fisiere transforms.json
├── models/             # Implementare MLP in PyTorch
├── src/                # Scripturi de training si volume rendering
│   ├── train.py        # Loop-ul de antrenare si evaluare
│   └── data_loader.py  # Preprocesare si calcul raze
├── outputs/            # Randari live din timpul antrenarii
├── README.md           # Documentatie clara si completa
└── requirements.txt    # Dependente Python (PyTorch, OpenCV, etc.)
