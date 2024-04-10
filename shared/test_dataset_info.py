from unittest import TestCase

from shared.dataset_info import name_to_ntu_data, name_to_ut_data


class Test(TestCase):
    def test_name_to_ut_data(self):
        path = "/tmp/test/0_1_4.avi"
        data = name_to_ut_data(path)
        info = data.info
        assert info == {'subject': 0, 'camera': 1, 'action': 4}

    def test_name_to_ntu_data_avi(self):
        path = "/media/barny/SSD4/MasterThesis/Data/nturgb+d_rgb/S017C003P020R002A060_rgb.avi"
        data = name_to_ntu_data(path)
        info = data.info
        assert info == {'set': 17, 'camera': 3, 'person': 20, 'replication': 2, 'action': 60}

    def test_name_to_ntu_data_skeleton(self):
        path = "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco/S017C003P020R002A060.coco17.apskel.pkl"
        data = name_to_ntu_data(path)
        info = data.info
        assert info == {'set': 17, 'camera': 3, 'person': 20, 'replication': 2, 'action': 60}
