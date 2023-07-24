train_txt = None

from train_and_test.dpp_trainer import BinarySegmentationDPP

config_dir = 'config/binary_segmentation_pretrain_4_5_100epoch.yaml'
print(config_dir)
bsdpp = BinarySegmentationDPP(train_txt, config_dir)
bsdpp.train_dpp()
