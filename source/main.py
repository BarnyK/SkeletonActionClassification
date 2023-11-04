import datasets
from procedures.preprocess_files import preprocess_files, PreprocessConfig
from procedures.training import train_network, TrainingConfig

# preprocess_files("/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
#                  "/media/barny/SSD4/MasterThesis/Data/ntu_coco.f1.combined", PreprocessConfig())
# preprocess_files("/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco",
#                  "/media/barny/SSD4/MasterThesis/Data/ntu_120_coco.f1.combined", PreprocessConfig())
# testing_generation()


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

if __name__ == "__main__2":
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
                      "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1",
                     PreprocessConfig(),
                     datasets.all_splits,
                     24,
                     False)

if __name__ == "__main__":
    norms = ["none", "mean_spine", "spine", "screen", "relative", "spine_align", "mean_spine_align"]
    for norm_type in norms:
        try:
            print(norm_type)
            cfg = TrainingConfig("xview_joints_" + norm_type, "stgcnpp", 80, "cuda:0", ["joints"], 64, 32,
                                 "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xview.train.pkl", 64,
                                 "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xview.test.pkl", 128, 8, 1,
                                 20, norm_type, 0.1, 0.9, 0.0002, True, 0, "logs/augment_test", True, 0.0, False)
            train_network(cfg)
        except FileExistsError as er:
            print(er)
