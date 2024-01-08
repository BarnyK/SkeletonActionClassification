prep_names=(filling_mice)
for t in ${prep_names[@]}; do
  python runner.py preprocess \
      ./configs/general/prep_tests/$t.yaml \
      /media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco \
      /media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco \
      --save-path /media/barny/SSD4/MasterThesis/Data/prepped_data/prep_tests/$t \
      --processes 24;
done