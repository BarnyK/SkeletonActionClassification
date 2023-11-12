import os.path
from argparse import Namespace

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
    visualize(data, video_file, 1000 // 30, window_name=window_name)


def handle_visualize(args: Namespace):
    if not os.path.isfile(args.skeleton_file):
        print(f"{args.skeleton_file} does not exist")
        return False
    if not os.path.isfile(args.video_file):
        print(f"{args.video_file} does not exist")
        return False
    visualize_skeleton(args.skeleton_file, args.video_file)


if __name__ == "__main__":
    visualize_skeleton("/home/barny/MasterThesis/Data/nturgb+d_skeletons/S008C002P030R001A037.skeleton",
                       "/home/barny/MasterThesis/Data/nturgb+d_rgb/S008C002P030R001A037_rgb.avi")
    visualize_skeleton("/home/barny/MasterThesis/Data/alphapose_skeletons/ut_set1_coco/0_1_4.coco17.apskel.pkl")
