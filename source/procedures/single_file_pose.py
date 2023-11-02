import os

import torch
from tqdm import tqdm

from pose_estimation import run_pose_worker, DetectionLoader, init_pose_model, init_detector, read_configs
from shared.dataset_info import DatasetInfo
from shared.structs import SkeletonData, FrameData


def single_file_pose(filename, device: torch.device, skeleton_type: str = "coco17"):
    assert os.path.isfile(filename)

    gcfg, dcfg, opts = read_configs(skeleton_type, device)

    detector = init_detector(opts, dcfg)
    detector.load_model()

    pose = init_pose_model(device, gcfg, gcfg.weights_file)
    det_loader = DetectionLoader(filename, detector, gcfg, opts, "video", 8, 64)
    det_loader.start()

    pose_data_queue = run_pose_worker(pose, det_loader, opts)
    tq = tqdm(range(det_loader.datalen), dynamic_ncols=True, disable=False)
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
        skeleton_type,
        DatasetInfo(),
        filename,
        det_loader.datalen,
        frames,
        len(frames),
        det_loader.frameSize,
    )
    # visualize(data, data.video_file, wait_key=1000//30, draw_bbox=True, draw_confidences=True, draw_frame_number=True)


if __name__ == "__main__":
    single_file_pose("/media/barny/SSD4/MasterThesis/Data/concatenated.avi", torch.device("cuda"), "coco17")
