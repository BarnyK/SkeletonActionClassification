import os.path
from argparse import Namespace

from pose_estimation.ntu_loader import read_file
from shared.structs import SkeletonData
from shared.visualize_skeleton_file import visualize


def visualize_skeleton(skeleton_file: str, video_file: str = None, window_name: str = "visualization",
                       save_file: str = "", **kwargs):
    if skeleton_file.endswith(".skeleton"):
        data = read_file(skeleton_file)
    else:
        data = SkeletonData.load(skeleton_file)
        if not video_file:
            video_file = data.video_file

    if video_file is None:
        raise ValueError("Video file not found")
    visualize(data, video_file, 1000 // 30, window_name=window_name, save_file=save_file, **kwargs)


def handle_visualize(args: Namespace):
    if not os.path.isfile(args.skeleton_file):
        print(f"{args.skeleton_file} does not exist")
        return False
    if not os.path.isfile(args.video_file):
        print(f"{args.video_file} does not exist")
        return False
    if args.save_file and not os.path.isdir(os.path.split(args.save_file)[0]):
        print(f"Folder for {args.save_file} does not exist")
        return False
    visualize_skeleton(args.skeleton_file, args.video_file, save_file=args.save_file, draw_bbox=args.draw_bbox)
