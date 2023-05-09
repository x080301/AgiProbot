from train_and_test.binary_segmentation import BinarySegmentation


binarysegmentation1 = BinarySegmentation(config_dir='config/binary_segmentation_4096points.yaml')
binarysegmentation1.train()

binarysegmentation2 = BinarySegmentation(config_dir='config/binary_segmentation_2048points.yaml')
binarysegmentation2.train()

binarysegmentation3 = BinarySegmentation(config_dir='config/binary_segmentation_1024points.yaml')
binarysegmentation3.train()

binarysegmentation4 = BinarySegmentation(config_dir='config/binary_segmentation_2048points_no_weights.yaml')
binarysegmentation4.train()
