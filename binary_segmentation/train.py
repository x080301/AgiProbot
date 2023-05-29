#
# from train_and_test.binary_segmentation import BinarySegmentation
#
# config_dir = 'config/binary_segmentation.yaml'
# print(config_dir)
# binarysegmentation = BinarySegmentation(config_dir=config_dir)
#
# binarysegmentation.train()

from train_and_test.dpp_trainer import BinarySegmentationDPP

if __name__ == "__main__":
    config_dir = 'config/binary_segmentation_head4.yaml'
    print(config_dir)
    bsdpp = BinarySegmentationDPP()
    bsdpp.train_dpp()
