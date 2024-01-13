import os

from procedures.config import TrainingConfig, GeneralConfig
from procedures.evaluate import evaluate_folder
from procedures.generate_alphapose_skeletons import gen_alphapose_skeletons
from procedures.preprocess_files import preprocess_files
from procedures.training import train_network
from shared.errors import DifferentConfigException

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

if __name__ == "__main__2":
    cfg = GeneralConfig()
    cfg.pose_config.dataset_name = "ut"
    input_folder = "/media/barny/SSD4/MasterThesis/Data/ut-interaction/my_segments1"
    output_folder = f"/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/my_ut_set1_coco"
    os.makedirs(output_folder, exist_ok=True)
    gen_alphapose_skeletons(input_folder, output_folder, cfg)

    cfg = GeneralConfig()
    cfg.pose_config.dataset_name = "ut"
    input_folder = "/media/barny/SSD4/MasterThesis/Data/ut-interaction/my_segments2"
    output_folder = f"/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/my_ut_set2_coco"
    os.makedirs(output_folder, exist_ok=True)
    gen_alphapose_skeletons(input_folder, output_folder, cfg)

    cfg = GeneralConfig.from_yaml_file("./configs/general/ut_test_conf.yaml")
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/my_ut_set1_coco",
                      "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/my_ut_set2_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_my_ut_test", cfg.prep_config)


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

if __name__ == "__main__2":
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
        ntu_files = [ntu_files[0]]

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

    ntu_files = ["/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_test2/ntu_mutual_xview.train.pkl",
                 "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_test2/ntu_mutual_xsub.train.pkl",
                 "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_test2/ntu120_mutual_xsub.train.pkl",
                 "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_test2/ntu120_mutual_xset.train.pkl"]
    for i in range(3):
        for file in ntu_files:
            name = os.path.split(file)[-1].split(".")[0]
            test_file = file.replace("train", "test")
            cfg = GeneralConfig.from_yaml_file("configs/general/2pgcn_ntu_xview.yaml")
            cfg.name = f"2pgcn_ntu_b32_sym_{name}_{i}"
            cfg.train_config.train_file = file
            cfg.eval_config.test_file = test_file
            cfg.symmetry_processing = True
            cfg.train_config.train_batch_size = 32
            cfg.eval_config.test_batch_size = 64
            train_network(cfg)
            torch.cuda.empty_cache()


def shortfeat(feat):
    return "".join(x[:2] for x in feat.split("_"))


if __name__ == "__main__2":
    cfg = GeneralConfig.from_yaml_file("./configs/general/ut_test_conf.yaml")
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ut_set1_coco",
                      "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ut_set2_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_ut_test2", cfg.prep_config)
if __name__ == "__main__2":
    config = GeneralConfig.from_yaml_file("configs/general/ut_test_conf.yaml")
    train_network(config)

    config = GeneralConfig.from_yaml_file("configs/general/ut_2pgcn_conf.yaml")
    train_network(config)

    config = GeneralConfig.from_yaml_file("configs/general/ut_2pgcn_conf.yaml")
    config.name = "2pgcn_ut_both_jo-jomo_b16_64"
    config.samples_per_window = 64
    train_network(config)

    config = GeneralConfig.from_yaml_file("configs/general/ut_2pgcn_conf.yaml")
    config.name = "2pgcn_ut_both_jo-jomo_b16_inter_spatial"
    config.graph_type = "mutual-inter"
    config.labeling = "spatial"
    train_network(config)

    config = GeneralConfig.from_yaml_file("configs/general/ut_2pgcn_conf.yaml")
    config.name = "2pgcn_ut_both_jo-jomo_b16_inter_spatial_copy_pad"
    config.graph_type = "mutual-inter"
    config.labeling = "spatial"
    config.copy_pad = True
    train_network(config)

    config = GeneralConfig.from_yaml_file("configs/general/ut_2pgcn_conf.yaml")
    config.train_config.train_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_my_ut_test/ut_whole_test.test.pkl"
    config.eval_config.test_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_my_ut_test/ut_whole_train.train.pkl"
    config.name = "2pgcn_myut_both_jo-jomo_b16_inter_spatial"
    config.graph_type = "mutual-inter"
    config.labeling = "spatial"
    train_network(config)

if __name__ == "__main__2":
    config = GeneralConfig.from_yaml_file("configs/general/ap_mutual_120xset.yaml")
    train_network(config)
    config = GeneralConfig.from_yaml_file("configs/general/ap_2pgcn_mutual_xview.yaml")
    train_network(config)
    config = GeneralConfig.from_yaml_file("configs/general/ap_2pgcn_mutual_120xset.yaml")
    train_network(config)

    config = GeneralConfig.from_yaml_file("configs/general/ap_2pgcn_mutual_120xset.yaml")
    config.name = config.name + "_noalign"
    config.normalization_type = "spine"
    train_network(config)

    config = GeneralConfig.from_yaml_file("configs/general/ap_2pgcn_mutual_120xset.yaml")
    config.name = config.name + "_copypad"
    config.copy_pad = True
    train_network(config)

    config = GeneralConfig.from_yaml_file("configs/general/ap_2pgcn_mutual_xview.yaml")
    config.name = config.name + "_noalign"
    config.normalization_type = "spine"
    train_network(config)

    config = GeneralConfig.from_yaml_file("configs/general/ap_2pgcn_mutual_xview.yaml")
    config.name = config.name + "_copypad"
    config.copy_pad = True
    train_network(config)
    # raise ValueError

if __name__ == "__main__2":
    log_folder = "/media/barny/SSD4/MasterThesis/Data/logs/transforms/"
    ap_config = GeneralConfig.from_yaml_file("configs/general/default_ap_transformed_xview.yaml")
    ap_config.log_folder = log_folder
    ap_config.train_config.train_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_transformed/ntu120_xset.train.pkl"
    ap_config.eval_config.test_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_transformed/ntu120_xset.test.pkl"
    ap_config.name = "ap_transformed_120xset"
    train_network(ap_config)

    ntu_config = GeneralConfig.from_yaml_file("configs/general/ntu_transform_xview.yaml")
    ntu_config.log_folder = log_folder
    ntu_config.train_config.train_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_transformed/ntu120_xset.train.pkl"
    ntu_config.eval_config.test_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_transformed/ntu120_xset.test.pkl"
    ntu_config.name = "ntu_transformed_120xset"
    train_network(ntu_config)

    # print("120XSET")
    # print("AP config")
    # evaluate(ap_config)
    # print("AP config on ntu")
    # evaluate(ap_config, None, None, ntu_config.eval_config.test_file)
    #
    # print("NTU config")
    # evaluate(ntu_config)
    # print("NTU config on ap")
    # evaluate(ntu_config, None, None, ap_config.eval_config.test_file)

    log_folder = "/media/barny/SSD4/MasterThesis/Data/logs/transforms/"
    ap_config = GeneralConfig.from_yaml_file("configs/general/default_ap_transformed_xsub.yaml")
    ap_config.log_folder = log_folder
    ap_config.train_config.train_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_transformed/ntu120_xsub.train.pkl"
    ap_config.eval_config.test_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_transformed/ntu120_xsub.test.pkl"
    ap_config.name = "ap_transformed_120xsub"
    train_network(ap_config)

    ntu_config = GeneralConfig.from_yaml_file("configs/general/ntu_transform_xsub.yaml")
    ntu_config.log_folder = log_folder
    ntu_config.train_config.train_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_transformed/ntu120_xsub.train.pkl"
    ntu_config.eval_config.test_file = "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_transformed/ntu120_xsub.test.pkl"
    ntu_config.name = "ntu_transformed_120xsub"
    train_network(ntu_config)

    # print("120XSUB")
    # print("AP config")
    # evaluate(ap_config)
    # print("AP config on ntu")
    # evaluate(ap_config, None, None, ntu_config.eval_config.test_file)
    #
    # print("NTU config")
    # evaluate(ntu_config)
    # print("NTU config on ap")
    # evaluate(ntu_config, None, None, ap_config.eval_config.test_file)

if __name__ == "__main__2":
    log_folder = "/media/barny/SSD4/MasterThesis/Data/logs/transforms/"
    ap_config = GeneralConfig.from_yaml_file("configs/general/default_ap_transformed_xview.yaml")
    ap_config.log_folder = log_folder
    ap_config.name = "ap_transformed_xview"
    train_network(ap_config)

    ntu_config = GeneralConfig.from_yaml_file("configs/general/ntu_transform_xview.yaml")
    ntu_config.log_folder = log_folder
    ntu_config.name = "ntu_transformed_xview"
    train_network(ntu_config)

    # print("XVIEW")
    # print("AP config")
    # evaluate(ap_config)
    # print("AP config on ntu")
    # evaluate(ap_config, None, None, ntu_config.eval_config.test_file)
    #
    # print("NTU config")
    # evaluate(ntu_config)
    # print("NTU config on ap")
    # evaluate(ntu_config, None, None, ap_config.eval_config.test_file)

    log_folder = "/media/barny/SSD4/MasterThesis/Data/logs/transforms/"
    ap_config = GeneralConfig.from_yaml_file("configs/general/default_ap_transformed_xsub.yaml")
    ap_config.log_folder = log_folder
    ap_config.name = "ap_transformed_xsub"
    train_network(ap_config)

    ntu_config = GeneralConfig.from_yaml_file("configs/general/ntu_transform_xsub.yaml")
    ntu_config.log_folder = log_folder
    ntu_config.name = "ntu_transformed_xsub"
    train_network(ntu_config)

    # print("XSUB")
    # print("AP config")
    # evaluate(ap_config)
    # print("AP config on ntu")
    # evaluate(ap_config, None, None, ntu_config.eval_config.test_file)
    #
    # print("NTU config")
    # evaluate(ntu_config)
    # print("NTU config on ap")
    # evaluate(ntu_config, None, None, ap_config.eval_config.test_file)

if __name__ == "__main__2":
    ap_config = GeneralConfig.from_yaml_file("configs/general/default_ap_xview.yaml")
    ap_config.log_folder = "/tmp/"
    ap_config.name = "test123"
    # train_network(ap_config)

if __name__ == "__main__2":
    evaluate_folder("/home/barny/MasterThesis/Data/logs/feature_test")

if __name__ == "__main__2":
    import itertools

    files = ["/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_test1/ntu_xview.train.pkl",
             "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_test1/ntu_xsub.train.pkl",
             "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_test1/ntu120_xsub.train.pkl",
             "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_test1/ntu120_xset.train.pkl"]

    all_features = ['joints', 'joint_motion', 'angles', 'joints_relative', 'joint_accel', 'bones', 'bone_motion',
                    'bone_angles', 'bone_accel']
    combs = [list(comb) for comb in itertools.combinations(all_features, 2)]
    for i in range(1):
        for file in files[:2]:
            for features in combs:
                feat_shorted = "-".join([shortfeat(x) for x in features])
                name = os.path.split(file)[-1].split(".")[0]
                test_file = file.replace("train", "test")
                cfg = GeneralConfig.from_yaml_file("configs/general/default_ap_xview.yaml")
                cfg.log_folder = "/media/barny/SSD4/MasterThesis/Data/logs/feature_test"
                cfg.features = features
                cfg.name = f"stgcn_{feat_shorted}_{name}_{i}"
                cfg.train_config.train_file = file
                cfg.eval_config.test_file = test_file
                train_network(cfg)
                torch.cuda.empty_cache()

    all_features = ['joints', 'joint_motion', 'angles', 'angles_motion', 'joints_relative', 'joint_accel', 'bones',
                    'bone_motion', 'bone_angles', 'bone_accel']
    combs = [list(comb) for comb in itertools.combinations(all_features, 1)]
    for i in range(1):
        for file in files:
            for features in combs:
                feat_shorted = "-".join([shortfeat(x) for x in features])
                name = os.path.split(file)[-1].split(".")[0]
                test_file = file.replace("train", "test")
                cfg = GeneralConfig.from_yaml_file("configs/general/default_ap_xview.yaml")
                cfg.log_folder = "/media/barny/SSD4/MasterThesis/Data/logs/feature_test"
                cfg.features = features
                cfg.name = f"stgcn_{feat_shorted}_{name}_{i}"
                cfg.train_config.train_file = file
                cfg.eval_config.test_file = test_file
                train_network(cfg)
                torch.cuda.empty_cache()

    all_features = ['joints', 'joint_motion', 'angles', 'joints_relative', 'joint_accel', 'bones', 'bone_motion',
                    'bone_angles', 'bone_accel']
    files = [files[1]]
    combs = [list(comb) for comb in itertools.combinations(all_features, 3)]
    for i in range(2):
        for file in files:
            for features in combs:
                feat_shorted = "-".join([shortfeat(x) for x in features])
                name = os.path.split(file)[-1].split(".")[0]
                test_file = file.replace("train", "test")
                cfg = GeneralConfig.from_yaml_file("configs/general/default_ap_xview.yaml")
                cfg.log_folder = "/media/barny/SSD4/MasterThesis/Data/logs/feature_test"
                cfg.features = features
                cfg.name = f"stgcn_{feat_shorted}_{name}_{i}"
                cfg.train_config.train_file = file
                cfg.eval_config.test_file = test_file
                cfg.eval_config.eval_interval = 80
                cfg.eval_config.eval_last_n = 10
                train_network(cfg)
                torch.cuda.empty_cache()

if __name__ == "__main__":
    norm_types = ["none", "spine_align", "mean_spine_align", "screen", "relative", "spine", "mean_spine"]

    for i in range(3):
        for sym in [False, True]:
            for labeling in ["distance", "spatial"]:
                for graph_type in ["mutual", "mutual-inter"]:
                    config = GeneralConfig.from_yaml_file("configs/general/norm_tests/2pgcn_120xsub_spine_align.yaml")
                    config.log_folder = "/media/barny/SSD4/MasterThesis/Data/logs/labeling_types/"
                    config.name = f"2pgcn_{labeling}_{graph_type}_{str(sym).lower()}_{i}"
                    config.labeling = labeling
                    config.graph_type = graph_type
                    config.symmetry_processing = sym
                    train_network(config)

    for i in range(1):
        for norm_type in norm_types:
            config = GeneralConfig.from_yaml_file("configs/general/norm_tests/st_120xsub_spine_align.yaml")
            config.name = config.name + f"_{norm_type}_{i}"
            config.normalization_type = norm_type
            train_network(config)

        for norm_type in norm_types:
            config = GeneralConfig.from_yaml_file("configs/general/norm_tests/2pgcn_120xsub_spine_align.yaml")
            config.name = config.name + f"_{norm_type}_{i}"
            config.normalization_type = norm_type
            train_network(config)

# if __name__ == "__main__2":
#     for sym in [False, True]:
#         for labeling in ["distance", "spatial"]:
#             for graph_type in ["mutual", "mutual-inter"]:
#                 config = GeneralConfig.from_yaml_file("configs/general/norm_tests/2pgcn_120xsub_spine_align.yaml")
#                 config.log_folder = "/media/barny/SSD4/MasterThesis/Data/logs/random/"
#                 config.name = f"2pgcn_{labeling}_{graph_type}_{str(sym).lower()}"
#                 config.labeling = labeling
#                 config.graph_type = graph_type
#                 config.symmetry_processing = sym
#                 train_network(config)
#                 break

if __name__ == "__main__":
    cfgs = [  # "/home/barny/thesis/source/configs/general/prep_tests/default.yaml",
        "/home/barny/thesis/source/configs/general/prep_tests/filling_none.yaml",
        "/home/barny/thesis/source/configs/general/prep_tests/filling_zero.yaml",
        "/home/barny/thesis/source/configs/general/prep_tests/no_filters.yaml",
        "/home/barny/thesis/source/configs/general/prep_tests/no_nms.yaml",
        "/home/barny/thesis/source/configs/general/prep_tests/no_tracking_order.yaml",
        "/home/barny/thesis/source/configs/general/prep_tests/no_tracking_conf.yaml",
        "/home/barny/thesis/source/configs/general/prep_tests/no_tracking_size.yaml",
        # "/home/barny/thesis/source/configs/general/prep_tests/filling_mice.yaml",
    ]
    for i in range(3):
        for cfg_file in cfgs:
            cfg = GeneralConfig.from_yaml_file(cfg_file)
            cfg.name = f"{cfg.name}_{i}"
            try:
                train_network(cfg)
            except Exception as ex:
                print(ex)
                continue
