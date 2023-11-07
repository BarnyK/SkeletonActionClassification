import os
import sys
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from alphapose.models import builder
from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from alphapose.utils.transforms import heatmap_to_coord_simple
from detector.yolo_api import YOLODetector
from easydict import EasyDict
from tqdm import tqdm

from shared.helpers import update_config
from shared.structs import Body


class DetectionLoader:
    def __init__(
            self, input_source, detector, cfg, opt, mode="image", batch_size=1, queue_size=128
    ):
        self.cfg = cfg
        self.opt = opt
        self.mode = mode
        self.device = opt.device

        if mode == "image":
            self.img_dir = opt.inputpath
            self.imglist = [
                os.path.join(self.img_dir, im_name.rstrip("\n").rstrip("\r"))
                for im_name in input_source
            ]
            self.datalen = len(input_source)
        elif mode == "video":
            stream = cv2.VideoCapture(input_source)
            assert stream.isOpened(), "Cannot capture source"
            self.path = input_source
            self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
            self.fps = stream.get(cv2.CAP_PROP_FPS)
            self.frameSize = (
                int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            self.videoinfo = {
                "fourcc": self.fourcc,
                "fps": self.fps,
                "frameSize": self.frameSize,
            }
            stream.release()

        self.detector = detector
        self.batchSize = batch_size
        leftover = 0
        if (self.datalen) % batch_size:
            leftover = 1
        self.num_batches = self.datalen // batch_size + leftover

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        if cfg.DATA_PRESET.TYPE == "simple":
            pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
            self.transformation = SimpleTransform(
                pose_dataset,
                scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0,
                sigma=self._sigma,
                train=False,
                add_dpg=False,
                gpu_device=self.device,
            )
        elif cfg.DATA_PRESET.TYPE == "simple_smpl":
            # TODO: new features
            from easydict import EasyDict as edict

            dummpy_set = edict(
                {
                    "joint_pairs_17": None,
                    "joint_pairs_24": None,
                    "joint_pairs_29": None,
                    "bbox_3d_shape": (2.2, 2.2, 2.2),
                }
            )
            self.transformation = SimpleTransform3DSMPL(
                dummpy_set,
                scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2.2, 2.2),
                rot=cfg.DATASET.ROT_FACTOR,
                sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False,
                add_dpg=False,
                loss_type=cfg.LOSS["TYPE"],
            )

        # initialize the queue used to store frames
        """
        image_queue: the buffer storing pre-processed images for object detection
        det_queue: the buffer storing human detection results
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """
        if opt.sp:
            self._stopped = False
            self.image_queue = Queue(maxsize=queue_size)
            self.det_queue = Queue(maxsize=10 * queue_size)
            self.pose_queue = Queue(maxsize=10 * queue_size)
        else:
            self._stopped = mp.Value("b", False)
            self.image_queue = mp.Queue(maxsize=queue_size)
            self.det_queue = mp.Queue(maxsize=10 * queue_size)
            self.pose_queue = mp.Queue(maxsize=10 * queue_size)

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to pre process images for object detection
        if self.mode == "image":
            image_preprocess_worker = self.start_worker(self.image_preprocess)
        elif self.mode == "video":
            image_preprocess_worker = self.start_worker(self.frame_preprocess)
        else:
            raise ValueError(f"mode {self.mode} is not supported")
        # start a thread to detect human in images
        image_detection_worker = self.start_worker(self.image_detection)
        # start a thread to post process cropped human image for pose estimation
        image_postprocess_worker = self.start_worker(self.image_postprocess)

        return [
            image_preprocess_worker,
            image_detection_worker,
            image_postprocess_worker,
        ]

    def stop(self):
        # clear queues
        self.clear_queues()

    def terminate(self):
        if self.opt.sp:
            self._stopped = True
        else:
            self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.image_queue)
        self.clear(self.det_queue)
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def image_preprocess(self):
        for i in range(self.num_batches):
            imgs = []
            orig_imgs = []
            im_names = []
            im_dim_list = []
            for k in range(
                    i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)
            ):
                if self.stopped:
                    self.wait_and_put(self.image_queue, (None, None, None, None))
                    return
                im_name_k = self.imglist[k]

                # expected image shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(im_name_k)
                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)
                orig_img_k = cv2.cvtColor(
                    cv2.imread(im_name_k), cv2.COLOR_BGR2RGB
                )  # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
                im_dim_list_k = orig_img_k.shape[1], orig_img_k.shape[0]

                imgs.append(img_k)
                orig_imgs.append(orig_img_k)
                im_names.append(os.path.basename(im_name_k))
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Human Detection
                imgs = torch.cat(imgs)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                # im_dim_list_ = im_dim_list

            self.wait_and_put(
                self.image_queue, (imgs, orig_imgs, im_names, im_dim_list)
            )

    def frame_preprocess(self):
        # Frame preprocess opens the video stream and reads from it to create batches of images
        # The batches are sent to image_queue
        stream = cv2.VideoCapture(self.path)
        assert stream.isOpened(), "Cannot capture source"

        for i in range(self.num_batches):
            imgs = []
            orig_imgs = []
            im_names = []
            im_dim_list = []
            for k in range(
                    i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)
            ):
                (grabbed, frame) = stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed or self.stopped:
                    # put the rest pre-processed frames to the queue
                    if len(imgs) > 0:
                        with torch.no_grad():
                            # Record original image resolution
                            imgs = torch.cat(imgs)
                            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                        self.wait_and_put(
                            self.image_queue, (imgs, orig_imgs, im_names, im_dim_list)
                        )
                    self.wait_and_put(self.image_queue, (None, None, None, None))

                    sys.stdout.flush()
                    stream.release()
                    return

                # expected frame shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(frame)

                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)

                im_dim_list_k = frame.shape[1], frame.shape[0]

                imgs.append(img_k)
                orig_imgs.append(frame[:, :, ::-1])
                im_names.append(str(k) + ".jpg")
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Record original image resolution
                imgs = torch.cat(imgs)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                # im_dim_list_ = im_dim_list

            self.wait_and_put(
                self.image_queue, (imgs, orig_imgs, im_names, im_dim_list)
            )
        stream.release()
        tqdm.write("Finished frame processing")

    def image_detection(self):
        # Reads batches of images from image_queue and performs detection on them
        for i in range(self.num_batches):
            imgs, orig_imgs, im_names, im_dim_list = self.wait_and_get(self.image_queue)
            if imgs is None or self.stopped:
                self.wait_and_put(
                    self.det_queue, (None, None, None, None, None, None, None)
                )
                return

            with torch.no_grad():
                # pad useless images to fill a batch, else there will be a bug
                for pad_i in range(self.batchSize - len(imgs)):
                    imgs = torch.cat((imgs, torch.unsqueeze(imgs[0], dim=0)), 0)
                    im_dim_list = torch.cat(
                        (im_dim_list, torch.unsqueeze(im_dim_list[0], dim=0)), 0
                    )

                dets = self.detector.images_detection(imgs, im_dim_list)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(orig_imgs)):
                        self.wait_and_put(
                            self.det_queue,
                            (orig_imgs[k], im_names[k], None, None, None, None, None),
                        )
                    continue
                if isinstance(dets, np.ndarray):
                    dets = torch.from_numpy(dets)
                dets = dets.cpu()
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]
                if self.opt.tracking:
                    ids = dets[:, 6:7]
                else:
                    ids = torch.zeros(scores.shape)

            for k in range(len(orig_imgs)):
                boxes_k = boxes[dets[:, 0] == k]
                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    self.wait_and_put(
                        self.det_queue,
                        (orig_imgs[k], im_names[k], None, None, None, None, None),
                    )
                    continue
                inps = torch.zeros(boxes_k.size(0), 3, *self._input_size)
                cropped_boxes = torch.zeros(boxes_k.size(0), 4)

                self.wait_and_put(
                    self.det_queue,
                    (
                        orig_imgs[k],
                        im_names[k],
                        boxes_k,
                        scores[dets[:, 0] == k],
                        ids[dets[:, 0] == k],
                        inps,
                        cropped_boxes,
                    ),
                )
        self.wait_and_put(
            self.det_queue, (None, None, None, None, None, None, None)
        )
        tqdm.write("Finished detection")

    def image_postprocess(self):
        # Post process
        for i in range(self.datalen):
            with torch.no_grad():
                (
                    orig_img,
                    im_name,
                    boxes,
                    scores,
                    ids,
                    inps,
                    cropped_boxes,
                ) = self.wait_and_get(self.det_queue)
                if orig_img is None or self.stopped:
                    self.wait_and_put(
                        self.pose_queue, (None, None, None, None, None, None, None)
                    )
                    return
                if boxes is None or boxes.nelement() == 0:
                    self.wait_and_put(
                        self.pose_queue,
                        (None, orig_img, im_name, boxes, scores, ids, None),
                    )
                    continue
                # imght = orig_img.shape[0]
                # imgwidth = orig_img.shape[1]
                for i, box in enumerate(boxes):
                    inps[i], cropped_box = self.transformation.test_transform(
                        orig_img, box
                    )
                    cropped_boxes[i] = torch.FloatTensor(cropped_box)

                # inps, cropped_boxes = self.transformation.align_transform(orig_img, boxes)

                self.wait_and_put(
                    self.pose_queue,
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes),
                )
        tqdm.write("Finished post processing")

    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value

    @property
    def length(self):
        return self.datalen


def init_detector(opts: EasyDict, detector_cfg: EasyDict):
    detector = YOLODetector(detector_cfg, opts)
    return detector


def init_pose_model(device: torch.device, general_config: EasyDict, weights_file: str):
    pose_model = builder.build_sppe(
        general_config.MODEL, preset_cfg=general_config.DATA_PRESET
    )
    pose_model.load_state_dict(torch.load(weights_file, map_location=device))
    pose_model.to(device, non_blocking=True)
    pose_model.eval()
    return pose_model


def read_ap_configs(
        skeleton_type: str = "coco17", device: torch.device = torch.device("cuda:0")
):
    opts = EasyDict()
    opts.device = device
    opts.sp = True  # single process
    opts.tracking = False  # IDK YET TODO
    opts.gpus = "0"
    opts.gpus = (
        [int(i) for i in opts.gpus.split(",")]
        if torch.cuda.device_count() >= 1
        else [-1]
    )

    detector_cfg = EasyDict()
    detector_cfg.CONFIG = "./configs/detector/yolov3-spp.cfg"
    detector_cfg.WEIGHTS = "./weights/detector/yolov3-spp.weights"
    detector_cfg.INP_DIM = 608
    detector_cfg.NMS_THRES = 0.6
    detector_cfg.CONFIDENCE = 0.1
    detector_cfg.NUM_CLASSES = 80

    if skeleton_type == "coco17":
        general_config = update_config(
            "./configs/alphapose/256x192_res50_lr1e-3_1x.yaml"
        )
        general_config.weights_file = "./weights/alphapose/fast_res50_256x192.pth"
    elif skeleton_type == "halpe":
        general_config = update_config(
            "./configs/alphapose/256x192_res50_lr1e-3_1x.halpe.yaml"
        )
        general_config.weights_file = (
            "./weights/alphapose/halpe26_fast_res50_256x192.pth"
        )
    else:
        raise ValueError(f"invalid skeleton type {skeleton_type}")

    return general_config, detector_cfg, opts


def pose_worker(
        pose_model, det_loader: DetectionLoader, pose_queue: Queue, opts: EasyDict
):
    batch_size = 5
    tq = tqdm(range(det_loader.datalen), dynamic_ncols=True, disable=True)
    for i in tq:
        with torch.no_grad():
            (
                inps,
                orig_img,
                im_name,
                boxes,
                scores,
                ids,
                cropped_boxes,
            ) = det_loader.read()
            if orig_img is None:
                break
            if boxes is None or boxes.nelement() == 0:
                # Empty frame with no detection
                pose_queue.put([])
                continue

            inps = inps.to(opts.device, non_blocking=True)
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batch_size:
                leftover = 1
            num_batches = datalen // batch_size + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batch_size: min((j + 1) * batch_size, datalen)]
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)

            hm_size = [hm.size(2), hm.size(3)]
            bodies = []
            for j in range(hm.size(0)):
                bbox = cropped_boxes[j].tolist()
                pose_coord, pose_score = heatmap_to_coord_simple(
                    hm[j], bbox, hm_shape=hm_size, norm_type=None
                )
                body = Body(pose_coord, pose_score, boxes[j], scores[j])
                bodies.append(body)
            pose_queue.put(bodies)
    pose_queue.put(None)  # None indicates end of Queue


def run_pose_worker(pose_model, det_loader: DetectionLoader, opts: EasyDict):
    pose_queue = Queue(64)
    pose_worker_process = Thread(
        target=pose_worker, args=(pose_model, det_loader, pose_queue, opts)
    )
    pose_worker_process.start()
    return pose_queue


def window_worker(
        q: Queue, datalen: int, pose_data_queue: Queue, length: int, interlace: int
):
    window = []
    for i in range(datalen):
        data = pose_data_queue.get()
        window.append(data)
        if len(window) == length:
            q.put(window)
            window = window[:interlace]
    return window


def run_window_worker(
        datalen: int, pose_data_queue: Queue, length: int, interlace: int
):
    q = Queue(5)
    window_worker_thread = Thread(
        target=window_worker, args=(q, datalen, pose_data_queue, length, interlace)
    )
    window_worker_thread.start()
    return q

