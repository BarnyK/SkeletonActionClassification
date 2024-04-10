prep_names=(default filling_mice filling_none filling_zero no_filters no_nms no_tracking_conf no_tracking_order no_tracking_size)
for t in ${prep_names[@]}; do
  python runner.py preprocess \
      ./configs/general/prep_tests/$t.yaml \
      /media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco \
      /media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco \
      --save-path /media/barny/SSD4/MasterThesis/Data/prepped_data/prep_tests2/$t \
      --processes 24 >> ~/preprocess_results.txt;
done