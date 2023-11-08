import datasets
from procedures.config import PreprocessConfig, TrainingConfig, GeneralConfig
from procedures.preprocess_files import preprocess_files
from procedures.training import train_network

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
    single_file_pose("/media/barny/SSD4/MasterThesis/Data/concatenated.avi", torch.device("cuda"))

if __name__ == "__main__2":
    # Generation of NTU datasets from Alphapose skeletons
    # preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
    #                   "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
    #                  "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1",
    #                  PreprocessConfig(),
    #                  datasets.all_splits,
    #                  24,
    #                  False)
    cfg = PreprocessConfig()
    cfg.keypoint_fill_type = "mice"
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
                      "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_mice_fill_bad", cfg, datasets.all_splits, 4,
                     False)

    cfg = PreprocessConfig()
    cfg.keypoint_fill_type = "knn"
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
                      "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_knn_fill_bad", cfg, datasets.all_splits, 4,
                     False)

if __name__ == "__main__2":
    # Generation of NTU datasets from Alphapose skeleton while changing the skeleton type
    # cfg = PreprocessConfig()
    # cfg.transform_to_combined = True
    # preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
    #                   "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
    #                  "/media/barny/SSD4/MasterThesis/Data/prepped_data/coco_combined",
    #                  cfg,
    #                  datasets.all_splits,
    #                  24,
    #                  False)

    # cfg = ntu_preprocess_cfg()
    # cfg.remove_missing_from_file = True
    # preprocess_files(["/media/barny/SSD4/MasterThesis/Data/nturgb+d_skeletons",
    #                   "/media/barny/SSD4/MasterThesis/Data/nturgb+d_skeletons_120"],
    #                  "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_test1",
    #                  cfg,
    #                  datasets.all_splits, 6, False,
    #                  "/media/barny/SSD4/MasterThesis/Data/NTU_RGBD120_samples_with_missing_skeletons.txt")
    cfg = PreprocessConfig()
    cfg.remove_missing_from_file = True
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
                      "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_no_missing",
                     cfg,
                     datasets.all_splits,
                     10,
                     False,
                     "/media/barny/SSD4/MasterThesis/Data/NTU_RGBD120_samples_with_missing_skeletons.txt")

if __name__ == "__main__2":
    norms = ["none", "mean_spine", "spine", "screen", "relative", "spine_align", "mean_spine_align"]
    for norm_type in norms:
        try:
            print(norm_type)
            cfg = TrainingConfig("xview_joints_" + norm_type, "stgcnpp", 80, "cuda:0", ["joints"], 64, 32,
                                 "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xview.train.pkl", 64,
                                 "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xview.test.pkl", 128, 8, 1,
                                 20, norm_type, 0.1, 0.9, 0.0002, True, 0, "logs/augment_test2", True, 0.2, False)
            train_network(cfg)
        except FileExistsError as er:
            print(er)

if __name__ == "__main__2":
    sets = [
        ("mutual_xsub", "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_mutual_xsub.train.pkl",
         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_mutual_xsub.test.pkl"),
        ("mutual_xview", "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_mutual_xview.train.pkl",
         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_mutual_xview.test.pkl"),
        ("mutual120_xsub", "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu120_mutual_xsub.train.pkl",
         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu120_mutual_xsub.test.pkl"),
        ("mutual120_xset", "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu120_mutual_xset.train.pkl",
         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu120_mutual_xset.test.pkl"),
        ("ntu_xsub", "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.train.pkl",
         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.test.pkl"),
        ("ntu_xview", "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xview.train.pkl",
         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xview.test.pkl"),
        ("ntu120_xsub", "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu120_xsub.train.pkl",
         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu120_xsub.test.pkl"),
        ("ntu120_xset", "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu120_xset.train.pkl",
         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu120_xset.test.pkl"),
    ]
    for x, train_path, test_path in sets:
        try:
            if "mutual" in x:
                eval_int = 1
            else:
                eval_int = 5
            print(x)
            cfg = TrainingConfig(x + "_joints_spine_align", "stgcnpp", 80, "cuda:0", ["joints"], 64, 32,
                                 train_path, 64,
                                 test_path, 128, 8, eval_int,
                                 20, "spine_align", 0.1, 0.9, 0.0002, True, 0, "logs/differnt_sets", True, 0.1, False)
            print(cfg.to_yaml())
            train_network(cfg)
        except FileExistsError as er:
            print(er)


def find_divisors(N):
    return [i for i in range(1, N + 1) if N % i == 0]


if __name__ == "__main__":
    for i in range(3):
        for win_length in [64, 100, 60, 16, 32, 30]:
            cfg = GeneralConfig.from_yaml_file("configs/general/default.yaml")
            cfg.window_length = win_length
            for samples_per_win in find_divisors(win_length):
                cfg.name = f"default_{cfg.window_length}_{samples_per_win}_{i}"
                cfg.samples_per_window = samples_per_win
                print(cfg.to_yaml())
                print(cfg.name)
                train_network(cfg)

    cfg = GeneralConfig.from_yaml_file("configs/general/default.yaml")
    cfg.window_length = 64
    for i in range(3):
        for samples_per_win in [1, 2, 4, 8, 16, 32, 64]:
            cfg.name = f"default_{cfg.window_length}_{samples_per_win}_{i}"
            cfg.samples_per_window = samples_per_win
            print(cfg.to_yaml())
            train_network(cfg)

    cfg = GeneralConfig.from_yaml_file("configs/general/default.yaml")
    cfg.window_length = 100
    for i in range(3):
        for samples_per_win in [1, 2, 4, 5, 10, 20, 25, 50, 100]:
            cfg.name = f"default_{cfg.window_length}_{samples_per_win}_{i}"
            cfg.samples_per_window = samples_per_win
            print(cfg.to_yaml())
            train_network(cfg)

    cfg = GeneralConfig.from_yaml_file("configs/general/default.yaml")
    cfg.window_length = 32
    for i in range(3):
        for samples_per_win in [1, 2, 4, 8, 16, 32]:
            cfg.name = f"default_{cfg.window_length}_{samples_per_win}_{i}"
            cfg.samples_per_window = samples_per_win
            print(cfg.to_yaml())
            train_network(cfg)

    cfg = GeneralConfig.from_yaml_file("configs/general/default.yaml")
    cfg.window_length = 16
    for i in range(3):
        for samples_per_win in [1, 2, 4, 8, 16]:
            cfg.name = f"default_{cfg.window_length}_{samples_per_win}_{i}"
            cfg.samples_per_window = samples_per_win
            print(cfg.to_yaml())
            train_network(cfg)
