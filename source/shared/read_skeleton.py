from shared.ntu_loader import read_file as read_ntu
from shared.structs import SkeletonData


def read_skeleton(filename: str) -> SkeletonData:
    if filename.endswith(".apskel.pkl"):
        data = SkeletonData.load(filename)
    elif filename.endswith(".skeleton"):
        data = read_ntu(filename)
    else:
        raise ValueError("Not supported ")
    return data


# if __name__ == "__main__":
#     read_skeleton('/media/barny/SSD4/MasterThesis/Data/nturgb+d_skeletons/S001C001P001R001A001.skeleton')
