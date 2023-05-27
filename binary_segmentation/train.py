# from train_and_test.binary_segmentation_ddp import train_ddp
#
# from train_and_test.binary_segmentation import BinarySegmentation

# binarysegmentation = BinarySegmentation(config_dir='config/binary_segmentation.yaml')
# binarysegmentation.train()
# train_ddp()

from train_and_test.binary_segmentation import BinarySegmentation

config_dir = 'config/binary_segmentation.yaml'
print(config_dir)
binarysegmentation = BinarySegmentation(config_dir=config_dir)

binarysegmentation.train()
