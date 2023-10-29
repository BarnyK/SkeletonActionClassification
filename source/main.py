import datasets
from procedures.preprocess_files import preprocess_files, PreprocessConfig
from procedures.training import main_test

# preprocess_files("/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
#                  "/media/barny/SSD4/MasterThesis/Data/ntu_coco.f1.combined", PreprocessConfig())
# preprocess_files("/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco",
#                  "/media/barny/SSD4/MasterThesis/Data/ntu_120_coco.f1.combined", PreprocessConfig())
# testing_generation()
if __name__ == "__main__":
    main_test()

if __name__ == "__main__2":
    import torch
    from procedures.single_file_pose import single_file_pose

    # single_file_pose(
    #     "/media/barny/SSD4/MasterThesis/Data/ntu_sample/S001C001P001R001A010_rgb.avi",
    #     torch.device("cuda"),
    #     "coco17",
    # )
    single_file_pose(
        "/media/barny/SSD4/MasterThesis/Data/concatenated.avi",
        torch.device("cuda"),
        "coco17",
    )

if __name__ == "__main__3":
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
                      "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1",
                     PreprocessConfig(),
                     datasets.all_splits,
                     24,
                     False)
