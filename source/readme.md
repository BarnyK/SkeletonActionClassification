TODO:

- [x] Dataset
- [x] Training loop
- [x] Eval sampler (Modify it so that it can produce X amount of samples)
- [x] Evaluation loop
- [x] Support generation from ut
- [x] PreprocessConfig usage
    - [x] Add algorithms for body selection without motion (size/confidence)
- [x] TrainingConfig usage
- [x] Add scale augmentation
- [x] Add possibility of other normalizations
- [x] Add shoulder alignment
- [x] Graph creation for training stats
- [x] Support skeleton between ntu and coco with 15 joints
    - [x] Transform from ntu to this
    - [x] Transform from coco to this
- [x] Support other skeletons
    - [x] Port all functions that use information about skeleton type to one module
- [x] Support preprocessing from nturgb
- [x] Continue training(1. compare cfgs, 2. check epochs recorded in log, 3. load model)
    -  [x] Load model ensure works
- [x] Evaluation from loaded model
- [x] 2P-GCN
- [x] Single file + evaluation
- [x] Visualization with windowing
- [ ] Figure out a way to fill missing joints if all other are missing

GENERAL PIPELINE:

- Generate skeletons
- Preprocess skeletons
- Train
- Classify/Evaluate



Aktualne możliwości:
- Generacja szkieletów alphapose
  - Zamiana całego zbioru 
  - Szkielety Coco17 i połączenie Coco z szkieletami NTU
- Preprocessing szkieletów
  - Filtracja na podstawie pewności bbox i szkieletów
  - Parametric pose non maximum suppression
  - Śledzenie z ogranieczniem odległości
  - Wybór śledzonych szkieletów na podstawie długości sekwencji oraz miary ruchu punktów
  - Uzupełniania szkieletów - interpolacja, MICE, K-nn
- Feature extraction
  - Punkty
  - Relacja punktów do punktu centralnego szkieletu
  - Kości
  - Ruch (kości i punktów)
  - Przyśpieszenie 
  - Kąty przy stawach
  - Nachylenia kości do osi układu współrzędnych
- Klasyfikatory
  - ST-GCN++ - klasyfikacja akcji i interakcji
  - 2P-GCN - klasyfikacja interakcji
- Trening sieci
- Testy sieci z zapisami wyników
- Możliwość generacji wyników ensemble różnych wariantów sieci
- Klasyfikacja video przy użyciu oknowania
