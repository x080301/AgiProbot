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

    config_dir = 'config/pointnet.yaml'
    print(config_dir)

    from train_and_test.ddp_trainer_pipeline import train_ddp_func

    train_ddp_func(train_txt, config_dir, valid_motors=None)
