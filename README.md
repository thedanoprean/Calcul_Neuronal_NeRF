# DeepScene 360 - Reconstrucție 3D folosind NeRF

## 1. Definirea Problemei și a Obiectivului
Acest proiect vizează implementarea unei rețele neuronale de tip **Neural Radiance Fields (NeRF)** pentru a genera reprezentări 3D continue ale unei scene plecând de la un set limitat de imagini 2D. Obiectivul este sintetizarea de vederi noi din unghiuri neexplorate în dataset-ul de antrenare.

## 2. Dataset
Voi utiliza un dataset de tip:
* **Sursă:** 
* **Conținut:** Imagini RGB însoțite de parametrii de poziție ai camerei (extrase prin COLMAP). 

## 3. Arhitectură Propusă
Modelul se bazează pe un **Multi-Layer Perceptron (MLP)** care învață o funcție de mapare de la coordonate spatiale $(x, y, z)$ și direcție de vizualizare $(\theta, \phi)$ la densitate volumetrică $\sigma$ și culoare $RGB$.

## 4. Plan de Lucru (Calendar Evaluări)
* **S4-S5:** Validarea temei și structura inițială a repository-ului (20%).
* **S8-S9:** Implementarea pipeline-ului și primul experiment complet (20%).
* **S12-S14:** Rezultate finale, comparație cu baseline și documentație completă (60%).
