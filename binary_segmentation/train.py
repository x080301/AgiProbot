# from train_and_test.binary_segmentation_ddp import train_ddp
#
# from train_and_test.binary_segmentation import BinarySegmentation

# binarysegmentation = BinarySegmentation(config_dir='config/binary_segmentation.yaml')
# binarysegmentation.train()
# train_ddp()
from train_and_test.binary_segmentation_dft import BinarySegmentation

binarysegmentation = BinarySegmentation(config_dir='config/binary_segmentation_dft_0.yaml')

binarysegmentation.train()
