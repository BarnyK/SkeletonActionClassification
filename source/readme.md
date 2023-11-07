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
- [ ] Evaluation from loaded model
- [ ] 2P-GCN
- [ ] Fix for alignement in mutual (double the bodies up front)
- [ ] Single file + evaluation
- [ ] Visualization with windowing
- [ ] Figure out a way to fill missing joints if all other are missing

GENERAL PIPELINE:

- Generate skeletons
- Preprocess skeletons
- Train/Classify/Evaluate

