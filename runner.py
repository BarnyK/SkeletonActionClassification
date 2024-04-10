import argparse

from procedures.evaluate import handle_eval
from procedures.generate_alphapose_skeletons import handle_generate
from procedures.preprocess_files import handle_preprocess
from procedures.single_file_classification import handle_classify
from procedures.single_file_pose import handle_pose_estimation
from procedures.training import handle_training
from procedures.visualize_skeleton import handle_visualize

a = """
- generate
- preprocess
- train
- eval
- classify
- visualize
"""


def main():
    parser = argparse.ArgumentParser("Action classification")
    subparsers = parser.add_subparsers(dest='function')

    # Generate
    generate_parser = subparsers.add_parser("generate", help="Generate skeletons.")
    generate_parser.add_argument("config", help="Config file that will be used to generate skeletons.")
    generate_parser.add_argument("input_folder", help="Input folder with video files.")
    generate_parser.add_argument("output_folder", help="Output folder in which generated files will be saved.")

    # Preprocessing
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess skeletons.")
    preprocess_parser.add_argument("config", help="Config file that will be used to generate skeletons.")
    preprocess_parser.add_argument("inputs", nargs='+',
                                   help="Input files that will be processed. Either a list of directories or files. " +
                                        "If it's files the processed skeletons will be saved." +
                                        "If it's directories the splits from config will be saved")
    preprocess_parser.add_argument("--processes", type=int, default=-1, help="Number of processes to use")
    preprocess_parser.add_argument("--save-path", required=True, default="",
                                   help="Path to which the files should be saved")

    # Training
    train_parser = subparsers.add_parser("train", help="Train network.")
    train_parser.add_argument("config", help="Config file that will be used to generate skeletons.")

    # Evaluation
    eval_parser = subparsers.add_parser("eval", help="Evaluate using a preprocessed file.", aliases=["evaluate"])
    eval_parser.add_argument("config", help="Config file that will be used to generate skeletons.")
    eval_parser.add_argument("--model", default="",
                             help="Model to be used in evaluation. Should fit the config provided.")
    eval_parser.add_argument("--save-file", default="", help="Save results to file.")

    # Classification
    classify_parser = subparsers.add_parser("classify", help="Classify a video file.")
    classify_parser.add_argument("config", help="Config file that will be used to generate skeletons.")
    classify_parser.add_argument("video_file", help="Video for classification")
    classify_parser.add_argument("--model", default="",
                                 help="Model to be used in evaluation. Should fit the config provided.")
    classify_parser.add_argument("--method", choices=['mean', 'window'], default="window",
                                 help="Method for classification ")
    classify_parser.add_argument("--save-file", default="", help="Save video file to file")
    classify_parser.add_argument("--window-save-file", default="", help="Save window results to file")
    classify_parser.add_argument("--interlace", default=None, type=int, help="Overwrite interlace value from config")
    classify_parser.add_argument("--window-length", default=None, type=int,
                                 help="Overwrite window length value from config")
    classify_parser.add_argument("--samples", default=None, type=int,
                                 help="Overwrite samples per window value from config")

    # Visualizer
    visualize_parser = subparsers.add_parser("visualize", help="Visualize skeleton with video file.")
    visualize_parser.add_argument("skeleton_file", help="Input file. Can be NTU or Alphapose skeleton file")
    visualize_parser.add_argument("video_file", help="Video file that will be played with skeletons")
    visualize_parser.add_argument("--save-file", default="", help="Save output video to file")
    visualize_parser.add_argument("--draw-bbox", action="store_true", help="Whether to draw bounding boxes")

    # Single file pose estimation
    estimator = subparsers.add_parser("esetimation", help="Pose estimate video file.")
    estimator.add_argument("config", help="Config file that will be used")
    estimator.add_argument("video_file", help="Video file that will undergo pose estimation")
    estimator.add_argument("--save-video", default="", help="Save output video")
    estimator.add_argument("--save-skeleton", default="", help="Save output skeletons")

    args = parser.parse_args()

    if args.function == "generate":
        handle_generate(args)
    elif args.function == "preprocess":
        handle_preprocess(args)
    elif args.function == "train":
        handle_training(args)
    elif args.function == "eval":
        handle_eval(args)
    elif args.function == "classify":
        handle_classify(args)
    elif args.function == "visualize":
        handle_visualize(args)
    elif args.function == "estimator":
        handle_pose_estimation(args)


if __name__ == "__main__":
    main()
