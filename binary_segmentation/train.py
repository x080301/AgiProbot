#
# from train_and_test.binary_segmentation import BinarySegmentation
#
# config_dir = 'config/binary_segmentation.yaml'
# print(config_dir)
# binarysegmentation = BinarySegmentation(config_dir=config_dir)
#
# binarysegmentation.train()

from train_and_test.dpp_trainer import main

config_dir = 'config/binary_segmentation.yaml'
print(config_dir)
main()
