from train_and_test.binary_segmentation import BinarySegmentation

binarysegmentation = BinarySegmentation(config_dir='config/binary_segmentation_4096_fine_tune.yaml')
binarysegmentation.train()
