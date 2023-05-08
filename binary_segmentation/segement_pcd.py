from train_and_test.binary_segmentation import BinarySegmentation

if __name__ == '__main__':
    binarysegmentation = BinarySegmentation()
    binarysegmentation.load_trained_model(
        'D:/Jupyter/AgiProbot/binary_segmentation/outputs/2023_05_05_21_24/train_on_sim/checkpoints/0.7977111328497994best_m.pth')
    binarysegmentation.test('E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_npy', save_dir='outputs/2023_05_05_21_24/train_on_sim',
                            save_pcd_dir=None)
