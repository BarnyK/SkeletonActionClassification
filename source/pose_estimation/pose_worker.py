from queue import Queue
from threading import Thread

import torch
from alphapose.utils.transforms import heatmap_to_coord_simple
from easydict import EasyDict
from tqdm import tqdm

from pose_estimation import DetectionLoader
from shared.structs import Body


def pose_worker(
        pose_model, det_loader: DetectionLoader, pose_queue: Queue, opts: EasyDict, batch_size: int = 5
):
    tq = tqdm(range(det_loader.datalen), dynamic_ncols=True, disable=True)
    for i in tq:
        with torch.no_grad():
            (
                inps,
                boxes,
                scores,
                cropped_boxes,
            ) = det_loader.read()
            if scores is None:
                break
            if boxes is None or boxes.nelement() == 0:
                # Empty frame with no detection
                pose_queue.put([])
                continue
            #
            # ## Threshold boxes
            # to_remove = []
            # left = []
            # for i in range(inps.shape[0]):
            #     if scores[i] < 0.7:
            #         to_remove = [i] + to_remove
            #     else:
            #         left.append(i)
            #
            # inps = inps[left]
            # boxes = boxes[left]
            # scores = scores[left]
            # ids = ids[left]
            # cropped_boxes = cropped_boxes[left]
            # for i in to_remove:
            #     orig_img.

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


def run_pose_worker(pose_model, det_loader: DetectionLoader, opts: EasyDict, batch_size: int = 5, queue_size: int = 64):
    pose_queue = Queue(queue_size)
    pose_worker_process = Thread(
        target=pose_worker, args=(pose_model, det_loader, pose_queue, opts, batch_size)
    )
    pose_worker_process.start()
    return pose_queue
