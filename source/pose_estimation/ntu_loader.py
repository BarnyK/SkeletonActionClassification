import numpy as np

from shared.dataset_info import name_to_ntu_data
from shared.structs import SkeletonData, FrameData, Body


def read_file(file_path):
    namedata = name_to_ntu_data(file_path)

    with open(file_path, "r") as fr:
        frame_count = int(fr.readline())
        skeleton_data = SkeletonData("NTU RGB+D", "ntu", namedata, "", frame_count, [], frame_count, (1920, 1080))

        for frame in range(frame_count):
            body_count_in_frame = int(fr.readline())

            frame_data = FrameData(frame, body_count_in_frame, [])
            for person in range(body_count_in_frame):
                person_info = fr.readline().strip().split()
                tracker_id = int(person_info[0])
                joint_count = int(fr.readline())
                joints = [fr.readline().strip().split() for _ in range(joint_count)]
                pose_xy = np.array([x[5:7] for x in joints], dtype=np.float32).reshape(
                    (-1, 2)
                )
                pose_xyz = np.array([x[:3] for x in joints], dtype=np.float32).reshape(
                    (-1, 3)
                )

                # Handle NaNs
                pose_xy[np.isnan(pose_xy)] = 0
                pose_xyz[np.isnan(pose_xy)[:, 0]] = 0
                pose_conf = 1 * np.invert(np.isnan(pose_xy)[:, :1])

                body = Body(pose_xy, pose_conf, None, None, pose_xyz, tracker_id)
                frame_data.bodies.append(body)
            skeleton_data.frames.append(frame_data)

    # Fix tracker numbers
    different_ids = sorted({body.tid for frame in skeleton_data.frames for body in frame.bodies})
    swap_dict = {id: i for i, id in enumerate(different_ids)}
    for body in [body for frame in skeleton_data.frames for body in frame.bodies]:
        body.tid = swap_dict[body.tid]

    return skeleton_data

if __name__ == "__main__":
    xd = read_file(
        "/media/barny/SSD4/MasterThesis/Data/nturgb+d_skeletons/S001C001P001R001A052.skeleton"
    )
    del xd.frames
    print(xd.__dict__)
