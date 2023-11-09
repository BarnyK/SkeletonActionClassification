from shared.ntu_loader import read_file
from shared.structs import SkeletonData
from shared.visualize_skeleton_file import visualize


def visualize_skeleton(skeleton_file: str, video_file: str = None, window_name: str = "visualization"):
    if skeleton_file.endswith(".skeleton"):
        data = read_file(skeleton_file)
    else:
        data = SkeletonData.load(skeleton_file)
        video_file = data.video_file

    if video_file is None:
        raise ValueError("Video file not found")
    visualize(data, video_file, 0, window_name=window_name)


if __name__ == "__main__":
    # with open("/media/barny/SSD4/MasterThesis/Data/bad.files", "r") as f:
    #     data = f.read().split("\n")
    #     data = [x.split("\t") for x in data]
    #
    # data = [x for x in data if x[0] != "motion"]
    # for reason, file in tqdm(data):
    #     data = SkeletonData.load(file)
    #     tqdm.write(str(len([body for frame in data.frames for body in frame.bodies])))
    #     tqdm.write(str([body.boxConf.mean().data for frame in data.frames for body in frame.bodies]))
    #     tqdm.write(str(np.mean([body.poseConf.mean() for frame in data.frames for body in frame.bodies])))
    #
    #     # print(reason)
    #     if "motion" in reason:
    #         continue
    # visualize_skeleton(file, window_name=reason)
    # visualize_skeleton(f"/home/barny/MasterThesis/Data/alphapose_skeletons/ntu_coco/S001C001P001R001A025.coco17.apskel.pkl")
    # visualize_skeleton("/home/barny/MasterThesis/Data/nturgb+d_skeletons/S008C002P030R001A037.skeleton",
    #                    "/home/barny/MasterThesis/Data/nturgb+d_rgb/S008C002P030R001A037_rgb.avi")
    visualize_skeleton("/home/barny/MasterThesis/Data/alphapose_skeletons/ut_set1_coco/0_1_4.coco17.apskel.pkl")
    # visualize_skeleton("/home/barny/MasterThesis/Data/alphapose_skeletons/ntu_120_coco/S030C002P075R002A096.coco17.apskel.pkl")

    # visualize_skeleton("/home/barny/thesis/source/sample_files/S009C003P025R001A060.coco17.apskel.pkl")
    # '/media/barny/SSD4/MasterThesis/Data/nturgb+d_rgb/S001C003P003R002A032_rgb.avi'
