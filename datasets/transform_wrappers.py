from __future__ import annotations

from typing import Type, Union

import numpy as np

import preprocessing.feature_extraction as fe


class PoseTransform:
    name: str
    skeleton_type: str
    requires: list[Type[PoseTransform]]

    def __init__(self, skeleton_type: str, *args, **kwargs):
        self.skeleton_type = skeleton_type

    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    def calculate_channels(in_chan: int):
        return in_chan


class Joints(PoseTransform):
    name = "joints"
    requires = []

    def __init__(self, skeleton_type: str, *args, **kwargs):
        super().__init__(skeleton_type, *args, **kwargs)

    def __call__(self, features: dict[str, np.ndarray]):
        joints = features.get(FeatureNames.JOINTS)
        if joints is None:
            raise KeyError(f"{FeatureNames.JOINTS} feature needs to be calculated before {self.name} feature")
        return features


class ToBones(PoseTransform):
    name = "bones"
    requires = [Joints]

    def __init__(self, skeleton_type: str, *args, **kwargs):
        super().__init__(skeleton_type, *args, **kwargs)

    def __call__(self, features: dict[str, np.ndarray]):
        joints = features.get(FeatureNames.JOINTS)
        if joints is None:
            raise KeyError(f"{FeatureNames.JOINTS} feature needs to be calculated before {self.name} feature")
        features[self.name] = fe.joints_to_bones(joints, self.skeleton_type)
        return features


class ToJointMotion(PoseTransform):
    name: str = "joint_motion"
    requires = [Joints]

    def __init__(self, skeleton_type: str, *args, **kwargs):
        super().__init__(skeleton_type, *args, **kwargs)

    def __call__(self, features: dict[str, np.ndarray]):
        joints = features.get(FeatureNames.JOINTS)
        if joints is None:
            raise KeyError(f"{FeatureNames.JOINTS} feature needs to be calculated before {self.name} feature")
        features[self.name] = fe.to_motion(joints)
        return features


class ToBoneMotion(PoseTransform):
    name: str = "bone_motion"
    requires = [ToBones]

    def __init__(self, skeleton_type: str, *args, **kwargs):
        super().__init__(skeleton_type, *args, **kwargs)

    def __call__(self, features: dict[str, np.ndarray]):
        bones = features.get(FeatureNames.BONES)
        if bones is None:
            raise KeyError(f"{FeatureNames.BONES}  feature needs to be calculated before {self.name} feature")
        features[self.name] = fe.to_motion(bones)
        return features


class ToJointAccel(PoseTransform):
    name: str = "joint_accel"
    requires = [Joints]

    def __init__(self, skeleton_type: str, *args, **kwargs):
        super().__init__(skeleton_type, *args, **kwargs)

    def __call__(self, features: dict[str, np.ndarray]):
        joints = features.get(FeatureNames.JOINTS)
        if joints is None:
            raise KeyError(f"{FeatureNames.JOINTS} feature needs to be calculated before {self.name} feature")
        features[self.name] = fe.to_accel(joints)
        return features


class ToBoneAccel(PoseTransform):
    name: str = "bone_accel"
    requires = [ToBoneMotion]

    def __init__(self, skeleton_type: str, *args, **kwargs):
        super().__init__(skeleton_type, *args, **kwargs)

    def __call__(self, features: dict[str, np.ndarray]):
        bones = features.get(FeatureNames.BONES)
        if bones is None:
            raise KeyError(f"{FeatureNames.BONES}  feature needs to be calculated before {self.name} feature")

        features[self.name] = fe.to_accel(bones)
        return features


class ToBoneAngles(PoseTransform):
    name: str = "bone_angles"
    requires = [ToBones]

    def __init__(self, skeleton_type: str, *args, **kwargs):
        super().__init__(skeleton_type, *args, **kwargs)

    def __call__(self, features: dict[str, np.ndarray]):
        bones = features.get(FeatureNames.BONES)
        if bones is None:
            raise KeyError(f"{FeatureNames.BONES}  feature needs to be calculated before {self.name} feature")

        features[self.name] = fe.bone_angles(bones)
        return features


class ToAngles(PoseTransform):
    name: str = "angles"
    requires = [Joints]

    def __init__(self, skeleton_type: str, *args, **kwargs):
        super().__init__(skeleton_type, *args, **kwargs)

    def __call__(self, features: dict[str, np.ndarray]):
        joints = features.get(FeatureNames.JOINTS)
        if joints is None:
            raise KeyError(f"{FeatureNames.JOINTS} feature needs to be calculated before bone {self.name} feature")
        features[self.name] = fe.to_angles(joints, self.skeleton_type)
        return features

    @staticmethod
    def calculate_channels(in_chan: int):
        return 1


class ToAnglesMotion(PoseTransform):
    name: str = "angles_motion"
    requires = [ToAngles]

    def __init__(self, skeleton_type: str, *args, **kwargs):
        super().__init__(skeleton_type, *args, **kwargs)

    def __call__(self, features: dict[str, np.ndarray]):
        angles = features.get(FeatureNames.ANGLES)
        if angles is None:
            raise KeyError(f"{FeatureNames.ANGLES} feature needs to be calculated before bone {self.name} feature")
        features[self.name] = fe.to_motion(angles)
        return features

    @staticmethod
    def calculate_channels(in_chan: int):
        return 1


class ToRelativeJoints(PoseTransform):
    name: str = "joints_relative"
    requires = [Joints]

    def __init__(self, skeleton_type: str, *args, **kwargs):
        super().__init__(skeleton_type, *args, **kwargs)

    def __call__(self, features: dict[str, np.ndarray]):
        joints = features.get(FeatureNames.JOINTS)
        if joints is None:
            raise KeyError(f"{FeatureNames.JOINTS} feature needs to be calculated before bone {self.name} feature")
        features[self.name] = fe.relative_joints(joints, self.skeleton_type)
        return features


class FeatureNames:
    JOINTS = Joints.name
    BONES = ToBones.name
    JOINT_MOTION = ToJointMotion.name
    BONE_MOTION = ToBoneMotion.name
    JOINT_ACCEL = ToJointAccel.name
    BONE_ACCEL = ToBoneAccel.name
    BONE_ANGLES = ToBoneAngles.name
    ANGLES = ToAngles.name
    JOINT_RELATIVE = ToRelativeJoints.name


TransformsList: list[Type[PoseTransform]] = [Joints, ToJointMotion, ToAngles, ToRelativeJoints, ToJointAccel, ToBones,
                                             ToBoneMotion, ToBoneAngles, ToBoneAccel, ToAnglesMotion]

TransformsNameList = [t.name for t in TransformsList]

TransformsDict: dict[str, Type[PoseTransform]] = {
    Joints.name: Joints,  # 1
    ToJointMotion.name: ToJointMotion,  # 2
    ToAngles.name: ToAngles,  # 2
    ToAnglesMotion.name: ToAnglesMotion,  # 3
    ToRelativeJoints.name: ToRelativeJoints,  # 2
    ToJointAccel.name: ToJointAccel,  # 3
    ToBones.name: ToBones,  # 2
    ToBoneMotion.name: ToBoneMotion,  # 3
    ToBoneAngles.name: ToBoneAngles,  # 3
    ToBoneAccel.name: ToBoneAccel,  # 4
}


def calculate_channels(features: Union[list[str], list[list[str], ...]], input_chan: int):
    if isinstance(features[0], list):
        features = features[0]
    channel_sum = 0
    for feat in features:
        channel_sum += TransformsDict[feat].calculate_channels(input_chan)
    return channel_sum
