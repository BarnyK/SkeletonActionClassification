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
- [ ] Support preprocessing from nturgb
- [ ] 2P-GCN
- [ ] Fix for alignement in mutual (double the bodies up front)
- [ ] Single file + evaluation
- [ ] Visualization with windowing

GENERAL PIPELINE:

- Generate skeletons
- Preprocess skeletons
- Train/Classify/Evaluate

