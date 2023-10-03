#
# from train_and_test.binary_segmentation import BinarySegmentation
#
# config_dir = 'config/binary_segmentation.yaml'
# print(config_dir)
# binarysegmentation = BinarySegmentation(config_dir=config_dir)
#
# binarysegmentation.train()

# from train_and_test.dpp_trainer import BinarySegmentationDPP
# from train_and_test.dpp_trainer_freeze import BinarySegmentationDPP


train_txt = None

if __name__ == "__main__":
    from train_and_test.dpp_trainer_pipeline import SegmentationDPP

    config_dir = 'config/4_6_200epoch_cross_validation_2,4.yaml'
    print(config_dir)
    bsdpp = SegmentationDPP(train_txt, config_dir)
    bsdpp.train_dpp()
