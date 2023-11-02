def single_file_classification(filename, device: torch.device, skeleton_type: str):
    assert os.path.isfile(filename)

    gcfg, dcfg, opts = read_configs(skeleton_type, device)

    detector = init_detector(opts, dcfg)
    detector.load_model()

    pose = init_pose_model(device, gcfg, gcfg.weights_file)
    det_loader = DetectionLoader(filename, detector, gcfg, opts, "video", 8, 256)
    det_loader.start()

    pose_data_queue = run_pose_worker(pose, det_loader, opts)

    length, interlace = 6, 3
    step = length - interlace
    no_windows = (det_loader.datalen - length) / step + 1

    wq = run_window_worker(det_loader.datalen, pose_data_queue, length, interlace)

    tq = tqdm(range(int(no_windows)), dynamic_ncols=True, disable=False)
    for i in tq:
        window = wq.get()
        print(window[0])
        # Do processing
        # Do classification
