import argparse

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
    parser.add_argument("--config", default="", help="Path to config file.")
    subparsers = parser.add_subparsers(dest='function')

    # Generate
    generate_parser = subparsers.add_parser("generate", help="Generate skeletons.")
    generate_parser.add_argument("config", help="Config file that will be used to generate skeletons.")
    generate_parser.add_argument("input_folder", help="Input folder with video files.")
    generate_parser.add_argument("output_folder", help="Output folder in which generated files will be saved.")

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess skeletons.")
    preprocess_parser.add_argument("config", help="Config file that will be used to generate skeletons.")
    preprocess_parser.add_argument("inputs", nargs='+',
                                   help="Input files that will be processed. Either a list of directories or files. " +
                                        "If it's files the processed skeletons will be saved." +
                                        "If it's directories the splits from config will be saved")
    preprocess_parser.add_argument("--processes", default=6, help="Number of processes to use")
    preprocess_parser.add_argument("--save-path", default="", help="Path to which the files should be saved")

    train_parser = subparsers.add_parser("train", help="Train network.")
    train_parser.add_argument("config", help="Config file that will be used to generate skeletons.")
    #TODO

    eval_parser = subparsers.add_parser("eval", help="Evaluate using a preprocessed file.")
    eval_parser.add_argument("config", help="Config file that will be used to generate skeletons.")
    eval_parser.add_argument("--model", default="",
                             help="Model to be used in evaluation. Should fit the config provided.")

    classify_parser = subparsers.add_parser("classify", help="Classify a video file.")
    classify_parser.add_argument("config", help="Config file that will be used to generate skeletons.")
    classify_parser.add_argument("video_file", help="Video for classification")
    eval_parser.add_argument("--model", default="",
                             help="Model to be used in evaluation. Should fit the config provided.")
    eval_parser.add_argument("--method", choices=['mean', 'window'], default="window",
                             help="Method for classification ")

    visualize_parser = subparsers.add_parser("visualize", help="Visualize skeleton with video file.")
    visualize_parser.add_argument("skeleton_file", help="Input file. Can be NTU or Alphapose skeleton file")
    visualize_parser.add_argument("video_file", help="Vide file that will be played with skeletons")
    # TODO ARG
    visualize_parser.add_argument("--save", default="", help="Save video file")

    args = parser.parse_args()

    print(args)


if __name__ == "__main__":
    main()
