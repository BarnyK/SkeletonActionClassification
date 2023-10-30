import os

import torch
from tqdm import tqdm

from pose_estimation import read_configs, init_detector, init_pose_model, DetectionLoader, run_pose_worker
from shared.dataset_info import name_to_ntu_data, name_to_ut_data
from shared.structs import SkeletonData, FrameData


def gen_alphapose_skeletons(
        input_folder: str, output_folder: str, skeleton_type: str = "coco17", name_to_info=name_to_ntu_data
):
    """
    Generates Alphapose skeleton files from videos
    input_folder should contain video files
    output_folder will contain skeleton frames dictionaries pickled
    """
    assert os.path.isdir(input_folder)
    assert os.path.isdir(output_folder)
    files = os.listdir(input_folder)
    files = [os.path.join(input_folder, fn) for fn in files]

    device = torch.device("cuda:0")
    gcfg, dcfg, opts = read_configs(skeleton_type, device)

    detector = init_detector(opts, dcfg)
    detector.load_model()

    pose = init_pose_model(device, gcfg, gcfg.weights_file)

    for file_i, file in tqdm(enumerate(files), total=len(files)):
        try:
            if file_i % 100 == 0:
                torch.cuda.empty_cache()

            dataset_info = name_to_info(file)

            outfile = os.path.join(output_folder, dataset_info.to_filename() + f".{skeleton_type}.apskel.pkl")
            if os.path.exists(outfile):
                continue

            det_loader = DetectionLoader(file, detector, gcfg, opts, "video", 5, 256)
            det_loader.start()

            pose_data_queue = run_pose_worker(pose, det_loader, opts)

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
                skeleton_type,
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


def generate_alphapose_dataset(input_folder: str, output_folder: str, skeleton_type: str, dataset_name: str):
    name_info_func_map = {
        "ntu": name_to_ntu_data,
        "ut": name_to_ut_data,
    }
    name_func = name_info_func_map.get(dataset_name, None)
    if name_func is None:
        print(f"Not supported dataset {dataset_name}")
    gen_alphapose_skeletons(input_folder, output_folder, skeleton_type, name_func)

def testing():
    input1 = "/media/barny/SSD4/MasterThesis/Data/ut-interaction/segmented_set1/"
    input2 = "/media/barny/SSD4/MasterThesis/Data/ut-interaction/segmented_set2/"
    out

    pass
def testing_generation():
    import time
    folder = f"/tmp/{time.time()}/"
    os.mkdir(folder)
    st = time.time()
    gen_alphapose_skeletons(
        "/media/barny/SSD4/MasterThesis/Data/sample2/",
        folder,
        "halpe",
    )
    et = time.time()
    print(et - st)
    st = time.time()
    gen_alphapose_skeletons(
        "/media/barny/SSD4/MasterThesis/Data/sample2/",
        folder,
        "coco17",
    )
    et = time.time()
    print(et - st)
