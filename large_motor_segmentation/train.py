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
    from train_and_test.ddp_trainer_pipeline import train_ddp_func

    valid_motors = '02&04'
    config_dir = 'config/segmentation_fine_tune_4_6_200epoch.yaml'
    print(config_dir)
    train_ddp_func(train_txt, config_dir, valid_motors, local_training=True)
