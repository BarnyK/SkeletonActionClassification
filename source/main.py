import os

import shared.datasets
from procedures.config import PreprocessConfig, TrainingConfig, GeneralConfig
from procedures.generate_alphapose_skeletons import gen_alphapose_skeletons
from procedures.preprocess_files import preprocess_files
from procedures.training import train_network
from shared.errors import DifferentConfigException

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
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_mice_fill_bad", cfg,
                     shared.datasets.all_splits, 4,
                     False)

    cfg = PreprocessConfig()
    cfg.keypoint_fill_type = "knn"
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
                      "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_knn_fill_bad", cfg,
                     shared.datasets.all_splits, 4,
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
    cfg = GeneralConfig()
    cfg.remove_missing_from_file = True
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
                      "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_no_missing",
                     cfg,
                     shared.datasets.all_splits,
                     10,
                     False,
                     "/media/barny/SSD4/MasterThesis/Data/NTU_RGBD120_samples_with_missing_skeletons.txt")

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

if __name__ == "__main__2":
    cfg = GeneralConfig()
    cfg.pose_config.dataset_name = "ntu"
    input_folder = "/media/barny/SSD4/MasterThesis/Data/nturgb+d_rgb/"
    output_folder = f"/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco"
    os.makedirs(output_folder, exist_ok=True)
    gen_alphapose_skeletons(input_folder, output_folder, cfg)

    cfg = GeneralConfig()
    cfg.pose_config.dataset_name = "ntu"
    input_folder = "/media/barny/SSD4/MasterThesis/Data/nturgb+d_rgb_120/"
    output_folder = f"/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"
    os.makedirs(output_folder, exist_ok=True)
    gen_alphapose_skeletons(input_folder, output_folder, cfg)

    cfg = GeneralConfig()
    cfg.pose_config.dataset_name = "ut"
    input_folder = "/media/barny/SSD4/MasterThesis/Data/ut-interaction/segmented_set1"
    output_folder = f"/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ut_set1_coco"
    os.makedirs(output_folder, exist_ok=True)
    gen_alphapose_skeletons(input_folder, output_folder, cfg)

    cfg = GeneralConfig()
    cfg.pose_config.dataset_name = "ut"
    input_folder = "/media/barny/SSD4/MasterThesis/Data/ut-interaction/segmented_set2"
    output_folder = f"/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ut_set2_coco"
    os.makedirs(output_folder, exist_ok=True)
    gen_alphapose_skeletons(input_folder, output_folder, cfg)


def find_divisors(N):
    return [i for i in range(1, N + 1) if N % i == 0 and i > 2]


if __name__ == "__main__2":
    cfg = GeneralConfig.from_yaml_file("configs/general/default_ap_xview.yaml")
    cfg.train_config.train_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_test1/ntu_mutual_xsub.train.pkl"
    cfg.eval_config.test_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_test1/ntu_mutual_xsub.test.pkl"
    cfg.name = "mutual_test1"
    train_network(cfg)

if __name__ == "__main__2":
    import torch

    for i in range(2):
        cfg = GeneralConfig.from_yaml_file("configs/general/ntu_xview.yaml")
        cfg.name = cfg.name + f"_{i}"
        train_network(cfg)
        torch.cuda.empty_cache()

        cfg = GeneralConfig.from_yaml_file("configs/general/ntu_xsub.yaml")
        cfg.name = cfg.name + f"_{i}"
        train_network(cfg)
        torch.cuda.empty_cache()

        cfg = GeneralConfig.from_yaml_file("configs/general/ntu_transform_xview.yaml")
        cfg.name = cfg.name + f"_{i}"
        train_network(cfg)
        torch.cuda.empty_cache()

        cfg = GeneralConfig.from_yaml_file("configs/general/default_ap_transformed_xview.yaml")
        cfg.name = cfg.name + f"_{i}"
        train_network(cfg)
        torch.cuda.empty_cache()

    for i in range(2):
        try:
            cfg = GeneralConfig.from_yaml_file("configs/general/ntu_3d_xview.yaml")
            cfg.name = cfg.name + f"_{i}"
            train_network(cfg)
            torch.cuda.empty_cache()
        except Exception as ex:
            print(ex)

if __name__ == "__main__2":
    import torch

    for i in range(3):
        for win_length in [16, 30, 32, 50, 60, 64, 100]:
            try:
                cfg = GeneralConfig.from_yaml_file("configs/general/default_ap_xview.yaml")
                cfg.log_folder = "/media/barny/SSD4/MasterThesis/Data/logs/window_tests/"
                cfg.eval_config.output_path = "/media/barny/SSD4/MasterThesis/Data/logs/window_tests/"
                cfg.window_length = win_length
                for samples_per_win in find_divisors(win_length):
                    cfg.name = f"default_{cfg.window_length}_{samples_per_win}_{i}"
                    cfg.samples_per_window = samples_per_win
                    print(cfg.name)
                    train_network(cfg)
                    torch.cuda.empty_cache()
            except DifferentConfigException as ex:
                print(ex)

if __name__ == "__main__":
    import torch

    files = ["/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_test1/ntu_mutual_xview.train.pkl",
             "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_test1/ntu_mutual_xsub.train.pkl",
             "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_test1/ntu120_mutual_xsub.train.pkl",
             "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_test1/ntu120_mutual_xset.train.pkl"]
    ntu_files = ["/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_test2/ntu_mutual_xview.train.pkl",
                 "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_test2/ntu_mutual_xsub.train.pkl",
                 "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_test2/ntu120_mutual_xsub.train.pkl",
                 "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_test2/ntu120_mutual_xset.train.pkl"]
    for i in range(3):
        for file in ntu_files:
            name = os.path.split(file)[-1].split(".")[0]
            test_file = file.replace("train", "test")
            cfg = GeneralConfig.from_yaml_file("configs/general/2pgcn_ntu_xview.yaml")
            cfg.name = f"2pgcn_ntu_{name}_{i}"
            cfg.train_config.train_file = file
            cfg.eval_config.test_file = test_file
            train_network(cfg)
            torch.cuda.empty_cache()
        ntu_files = ntu_files[0]

        for file in files:
            name = os.path.split(file)[-1].split(".")[0]
            test_file = file.replace("train", "test")
            cfg = GeneralConfig.from_yaml_file("configs/general/2pgcn_ap_xview.yaml")
            cfg.name = f"2pgcn_{name}_{i}"
            cfg.train_config.train_file = file
            cfg.eval_config.test_file = test_file
            train_network(cfg)
            torch.cuda.empty_cache()

            name = os.path.split(file)[-1].split(".")[0]
            test_file = file.replace("train", "test")
            cfg = GeneralConfig.from_yaml_file("configs/general/2pgcn_ap_xview.yaml")
            cfg.symmetry_processing = True
            cfg.name = f"2pgcn_sym_{name}_{i}"
            cfg.train_config.train_file = file
            cfg.eval_config.test_file = test_file
            train_network(cfg)
            torch.cuda.empty_cache()

            name = os.path.split(file)[-1].split(".")[0]
            test_file = file.replace("train", "test")
            cfg = GeneralConfig.from_yaml_file("configs/general/2pgcn_ap_xview.yaml")
            cfg.labeling = "spatial"
            cfg.name = f"2pgcn_spatial_{name}_{i}"
            cfg.train_config.train_file = file
            cfg.eval_config.test_file = test_file
            train_network(cfg)
            torch.cuda.empty_cache()

            name = os.path.split(file)[-1].split(".")[0]
            test_file = file.replace("train", "test")
            cfg = GeneralConfig.from_yaml_file("configs/general/2pgcn_ap_xview.yaml")
            cfg.graph_type = "mutual-inter"
            cfg.name = f"2pgcn_inter_{name}_{i}"
            cfg.train_config.train_file = file
            cfg.eval_config.test_file = test_file
            train_network(cfg)
            torch.cuda.empty_cache()

            name = os.path.split(file)[-1].split(".")[0]
            test_file = file.replace("train", "test")
            cfg = GeneralConfig.from_yaml_file("configs/general/2pgcn_ap_xview.yaml")
            cfg.graph_type = "mutual-inter"
            cfg.labeling = "spatial"
            cfg.name = f"2pgcn_inter_spat_{name}_{i}"
            cfg.train_config.train_file = file
            cfg.eval_config.test_file = test_file
            train_network(cfg)
            torch.cuda.empty_cache()

            name = os.path.split(file)[-1].split(".")[0]
            test_file = file.replace("train", "test")
            cfg = GeneralConfig.from_yaml_file("configs/general/2pgcn_ap_xview.yaml")
            cfg.train_config.train_batch_size = 16
            cfg.eval_config.test_batch_size = 64
            cfg.name = f"2pgcn_b16_{name}_{i}"
            cfg.train_config.train_file = file
            cfg.eval_config.test_file = test_file
            train_network(cfg)
            torch.cuda.empty_cache()

            name = os.path.split(file)[-1].split(".")[0]
            test_file = file.replace("train", "test")
            cfg = GeneralConfig.from_yaml_file("configs/general/2pgcn_ap_xview.yaml")
            cfg.train_config.train_batch_size = 16
            cfg.eval_config.test_batch_size = 64
            cfg.train_config.epochs = 65
            cfg.name = f"2pgcn_e65_b16_{name}_{i}"
            cfg.train_config.train_file = file
            cfg.eval_config.test_file = test_file
            train_network(cfg)
            torch.cuda.empty_cache()

            name = os.path.split(file)[-1].split(".")[0]
            test_file = file.replace("train", "test")
            cfg = GeneralConfig.from_yaml_file("configs/general/2pgcn_ap_xview.yaml")
            cfg.train_config.train_batch_size = 32
            cfg.eval_config.test_batch_size = 64
            cfg.name = f"2pgcn_b32_{name}_{i}"
            cfg.train_config.train_file = file
            cfg.eval_config.test_file = test_file
            train_network(cfg)
            torch.cuda.empty_cache()
        files = [files[0]]


