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
    print(datasets.TransformsNameList)
    feats = ["joints", "joints_relative", "bones"]
    for feat in feats:
        feat_initials = "".join([x[:2] for x in feat.split("_")])
        print(feat_initials, feat)
        cfg = TrainingConfig("xsub_" + feat_initials, "stgcnpp", 80, "cuda:0", [feat], 64, 32,
                             "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.train.pkl", 64,
                             "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.test.pkl", 128, 8, 1,
                             0, "mean_spine", 0.1, 0.9, 0.0002, True, 0, "logs", False, 0)
        train_network(cfg)
