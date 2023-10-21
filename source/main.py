from procedures.preprocess_files import preprocess_files, PreprocessConfig

preprocess_files("/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
                 "/media/barny/SSD4/MasterThesis/Data/ntu_coco.2.combined", PreprocessConfig())
preprocess_files("/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco",
                 "/media/barny/SSD4/MasterThesis/Data/ntu_120_coco.2.combined", PreprocessConfig())
# testing_generation()
