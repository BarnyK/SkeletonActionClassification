import os

import torch
from tqdm import tqdm

from pose_estimation.detection_loader import DetectionLoader
from pose_estimation.helpers import init_detector, init_pose_model, read_ap_configs
from pose_estimation.pose_worker import run_pose_worker
from procedures.config import GeneralConfig
from shared.dataset_info import name_info_func_map
from shared.structs import SkeletonData, FrameData


def gen_alphapose_skeletons(
        input_folder: str,
        output_folder: str,
        cfg: GeneralConfig
):
    """
    Generates Alphapose skeleton files from videos
    input_folder should contain video files
    output_folder will contain skeleton frames dictionaries pickled
    """
    pose_cfg = cfg.pose_config
    name_to_info = name_info_func_map.get(pose_cfg.dataset_name, None)
    if name_to_info is None:
        print(f"Not supported dataset {pose_cfg.dataset_name}")
    assert os.path.isdir(input_folder)
    assert os.path.isdir(output_folder)
    files = os.listdir(input_folder)
    files = [os.path.join(input_folder, fn) for fn in files]

    device = torch.device(cfg.device)
    ap_cfg, det_cfg, opts = read_ap_configs(cfg.skeleton_type, device)

    detector = init_detector(opts, det_cfg)
    detector.load_model()

    pose = init_pose_model(device, ap_cfg, ap_cfg.weights_file)

    for file_i, file in tqdm(enumerate(files), total=len(files)):
        try:
            if file_i % 100 == 0:
                torch.cuda.empty_cache()

            dataset_info = name_to_info(file)

            outfile = os.path.join(output_folder,
                                   dataset_info.to_filename() + f".{cfg.skeleton_type}.apskel.pkl")
            if os.path.exists(outfile):
                continue

            det_loader = DetectionLoader(file, detector, ap_cfg, opts, "video", pose_cfg.detector_batch_size,
                                         pose_cfg.detector_queue_size, 1)
            det_loader.start()

            pose_data_queue = run_pose_worker(pose, det_loader, opts, pose_cfg.estimation_batch_size,
                                              pose_cfg.estimation_queue_size)

            tq = tqdm(range(det_loader.datalen), dynamic_ncols=True, disable=True)
            frames = []
            for i in tq:
                data = pose_data_queue.get()
                if data is None:
                    # End of file
                    break
                frame_data = FrameData(i, len(data), data)
                frames.append(frame_data)

            data = SkeletonData(
                "estimated",
                cfg.skeleton_type,
                dataset_info,
                file,
                det_loader.datalen,
                frames,
                len(frames),
                det_loader.frameSize,
            )
            data.save(outfile)
        except Exception as ex:
            print(file, file_i, ex)
            raise ex


def testing():
    import time
    # UT
    cfg = GeneralConfig()
    cfg.pose_config.dataset_name = "ut"
    input_folder = "/media/barny/SSD4/MasterThesis/Data/ut_sample/"
    output_folder = f"/tmp/{time.time()}/"
    os.mkdir(output_folder)
    gen_alphapose_skeletons(input_folder, output_folder, cfg)
    print(cfg.to_yaml())
    # NTU
    cfg.pose_config.dataset_name = "ntu"
    input_folder = "/media/barny/SSD4/MasterThesis/Data/ntu_sample/"
    gen_alphapose_skeletons(input_folder, output_folder, cfg)
    print(cfg.to_yaml())


if __name__ == "__main__":
    testing()
