#
# from train_and_test.binary_segmentation import BinarySegmentation
#
# config_dir = 'config/binary_segmentation.yaml'
# print(config_dir)
# binarysegmentation = BinarySegmentation(config_dir=config_dir)
#
# binarysegmentation.train()

# from train_and_test.dpp_trainer import BinarySegmentationDPP

from train_and_test.dpp_trainer_no_rotation_augment import BinarySegmentationDPP
import shutil

train_txt = None
if __name__ == "__main__":
    config_dir = 'config/binary_segmentation_5_6.yaml'
    print(config_dir)
    bsdpp = BinarySegmentationDPP(train_txt, config_dir)
    best_m_path = bsdpp.train_dpp()

    shutil.copyfile(best_m_path, 'best_m.pth')

    config_dir = 'config/binary_segmentation_fine_tune_6_8.yaml'
    print(config_dir)
    bsdpp = BinarySegmentationDPP(train_txt, config_dir)
    bsdpp.train_dpp()
