This is a dataset adaptor which converts lerobot dataset to rlds data format so we can use lerobot dataset for finetuning openVLA (https://github.com/moojink/openvla-oft)
The code gets inspiration from https://github.com/moojink/rlds_dataset_builder/tree/main
However the original code requires python<=3.9 which is imcompatible with lerobot (python >= 3.10)

Steps: 
(1) `python convert_lerobot_to_hdf5.py` in a lerobot conda environment 


(2) switch to rlds_dataset envoironment (see https://github.com/moojink/rlds_dataset_builder/tree/main) and run `tfds build --overwrite`

(3) If you want to test if the conversion is successful or not, you can run `python3 visualize_dataset.py <name_of_your_dataset>` from the rlds_dataset_builder repo
