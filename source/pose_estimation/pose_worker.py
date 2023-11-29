from __future__ import annotations

from queue import Queue
from threading import Thread

import torch
import torch.multiprocessing as mp
from alphapose.utils.transforms import heatmap_to_coord_simple
from easydict import EasyDict
from tqdm import tqdm

from pose_estimation import DetectionLoader
from shared.structs import Body

use_mp = False
if use_mp:
    mp.set_start_method('forkserver', force=True)
    mp.set_sharing_strategy('file_system')


class ResultStore:
    def __init__(self, seq_id, count, boxes, scores, cropped_boxes):
        self.seqId = seq_id
        self.results = []
        self.count = count
        self.boxes = boxes
        self.scores = scores
        self.cropped_boxes = cropped_boxes

    def finished(self):
        return len(self.results) >= self.count

    def to_bodies(self) -> list[Body]:
        if len(self.results) == 0:
            return []
        heatmap_size = self.results[0].shape[-2:]
        bodies = []
        for i, heatmap in enumerate(self.results):
            bbox = self.cropped_boxes[i].tolist()
            pose_coord, pose_score = heatmap_to_coord_simple(
                heatmap, bbox, hm_shape=heatmap_size, norm_type=None
            )
            body = Body(pose_coord, pose_score, self.boxes[i], self.scores[i])
            bodies.append(body)
        return bodies


def pose_worker_batch_filling(pose_model, det_loader: DetectionLoader, pose_queue: Queue, opts: EasyDict,
                              batch_size: int = 5):
    # Holds a list of tuples with frame_id, tensor
    data_list = []
    # Holds a map of frame_seq to its size of inputs
    results = {}
    with torch.no_grad():
        for frame_id in range(det_loader.datalen):
            (inputs, orig_img, boxes, scores, cropped_boxes) = det_loader.read()
            if orig_img is None:
                # send current batch
                break
            if boxes is None or boxes.nelement() == 0:
                # empty bodies idk how to deal with this
                results[frame_id] = ResultStore(frame_id, 0, [], [], [], )
                continue
            datalen = inputs.size(0)
            results[frame_id] = ResultStore(frame_id, datalen, boxes, scores, cropped_boxes, )
            data_list.extend([(frame_id, inp) for inp in inputs])

            while len(data_list) >= batch_size:
                current_batch = data_list[:batch_size]
                data_list = data_list[batch_size:]

                send_batch(current_batch, opts, pose_model, results)

            res_keys = sorted(results.keys())
            for key in res_keys:
                if not results[key].finished():
                    break
                data = results.pop(key)
                bodies = data.to_bodies()
                pose_queue.put(bodies)

        send_batch(data_list, opts, pose_model, results)
        res_keys = sorted(results.keys())
        for key in res_keys:
            data = results.pop(key)
            bodies = data.to_bodies()
            pose_queue.put(bodies)
    pose_queue.put(None)
    tqdm.write("Finished pose")
    pass


def send_batch(current_batch, opts, pose_model, results):
    tensors: list[torch.Tensor] = [x[1] for x in current_batch]
    frame_ids = [x[0] for x in current_batch]
    input_tensor = torch.stack(tensors, 0)
    input_tensor = input_tensor.to(opts.device, non_blocking=True)
    heat_maps = pose_model(input_tensor)
    heat_maps_list = [x for x in heat_maps]
    for j, fid in enumerate(frame_ids):
        results[fid].results.append(heat_maps_list[j])


def pose_worker(
        pose_model, det_loader: DetectionLoader, pose_queue: Queue, opts: EasyDict, batch_size: int = 5
):
    tq = tqdm(range(det_loader.datalen), dynamic_ncols=True, disable=True)
    for i in tq:
        with torch.no_grad():
            (
                inps,
                orig_img,
                boxes,
                scores,
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


def run_pose_worker(pose_model, det_loader: DetectionLoader, opts: EasyDict, batch_size: int = 5, queue_size: int = 64):
    if use_mp:
        return run_pose_worker_mp(pose_model, det_loader, opts, batch_size, queue_size)
    pose_queue = Queue(queue_size)
    pose_worker_thread = Thread(
        target=pose_worker, args=(pose_model, det_loader, pose_queue, opts, batch_size)
    )
    pose_worker_thread.start()
    return pose_queue, pose_worker_thread


def run_pose_worker_mp(pose_model, det_loader: DetectionLoader, opts: EasyDict, batch_size: int = 5,
                       queue_size: int = 64):
    pose_queue = mp.Queue(queue_size)
    pose_worker_process = mp.Process(
        target=pose_worker, args=(pose_model, det_loader, pose_queue, opts, batch_size)
    )
    pose_worker_process.start()
    return pose_queue
