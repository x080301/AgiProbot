"""
@Author: bixuelei
@Contact: bxueleibi@gmial.com
@File: train_semseg.py
@Time: 2022/1/10 7:49 PM
"""

#from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import *
from model import DGCNN_semseg ,DGCNN_patch_semseg
import numpy as np
from torch.utils.data import DataLoader
from util import normalize_data,rotate_per_batch,PrintLog
from tqdm import tqdm
from visualize import Visuell_PointCloud_accordding_to_label



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes = ['clamping_system', 'cover', 'gear_container', 'charger', 'bottom', 'side_screw','cover_screw']                                                                                                                               
labels2categories={i:cls for i,cls in enumerate(classes)}       #dictionary for labels2categories

      
def test(args):
    NUM_POINT=args.npoints
    print("start loading test data ...")
    TEST_DATASET = MotorDataset_patch(split='Validation', data_root=args.data_dir, num_points=NUM_POINT, screw_weight=args.screw_weight, test_area=args.test_symbol, sample_rate=1.0, transform=None)
    test_loader = DataLoader(TEST_DATASET, num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    io = PrintLog(BASE_DIR+'/predicted_result/'+TEST_DATASET.motor_name[0].split('.')[0]+'.log')

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    elif args.model == 'dgcnn_patch':
        model = DGCNN_patch_semseg(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    print("Let's test and use", torch.cuda.device_count(), "GPUs!")

    try:
        if args.model=="dgcnn":
            checkpoint = torch.load(str(BASE_DIR)+"/trained_model"+"/best_finetune.pth")
        else:
            checkpoint = torch.load(str(BASE_DIR)+"/trained_model"+"/best_finetune_patch.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print('No existing model, a trained model is needed')
        exit(-1)

    with torch.no_grad():         
        num_batches = len(test_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASS)
        total_seen_class = [0 for _ in range(NUM_CLASS)]
        total_correct_class = [0 for _ in range(NUM_CLASS)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASS)]
        noBG_seen_class = [0 for _ in range(NUM_CLASS-1)]
        noBG_correct_class = [0 for _ in range(NUM_CLASS-1)]
        noBG_iou_deno_class = [0 for _ in range(NUM_CLASS-1)]
        model=model.eval()

        for i,(data,seg) in tqdm(enumerate(test_loader),total=len(test_loader),smoothing=0.9):
            data, seg = data.to(device), seg.to(device)
            data=normalize_data(data)
            data_=data
            data,GT=rotate_per_batch(data,None)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred,trans,_,= model(data,seg)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()   #array(batch_size*num_points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASS)   # (batch_size*num_points , num_class)
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  #array(batch_size*num_points)
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (batch_size * NUM_POINT)
            tmp, _ = np.histogram(batch_label, range(NUM_CLASS + 1))
            labelweights += tmp
            if args.vis:
                if i==0:
                    points=data_.view(-1, 3).cpu().data.numpy() 
                    pred_choice_=np.reshape(pred_choice,(-1,1))
                    points=np.hstack((points,pred_choice_))
                else:
                    points_=data_.view(-1, 3).cpu().data.numpy() 
                    pred_choice_=np.reshape(pred_choice,(-1,1))
                    points_=np.hstack((points_,pred_choice_))
                    points=np.vstack((points,points_))

            for l in range(NUM_CLASS):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))

            ####### calculate without Background ##############
            for l in range(1, NUM_CLASS):
                noBG_seen_class[l-1] += np.sum((batch_label == l))
                noBG_correct_class[l-1] += np.sum((pred_choice == l) & (batch_label == l))
                noBG_iou_deno_class[l-1] += np.sum(((pred_choice == l) | (batch_label == l)))

        if args.vis:
            if not args.save:
                Visuell_PointCloud_accordding_to_label(points)
            else:
                save_path=BASE_DIR+'/predicted_result/'+TEST_DATASET.motor_name[0].split('.')[0]
                Visuell_PointCloud_accordding_to_label(points,SavePCDFile=1,FileName=save_path)


        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))

        outstr = 'Test with backgroud: mIoU %.6f,  Test point acc %.6f, Test avg class acc %.6f' % (mIoU,
                                                    (total_correct / float(total_seen)),(np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))

        io.cprint(outstr)
        noBG_mIoU = np.mean(np.array(noBG_correct_class) / (np.array(noBG_iou_deno_class, dtype=np.float64) + 1e-6))
        outstr_without_background='Test without backgroud----mIoU %.6f,  Test point acc: %.6f, Test avg class acc: %.6f' % (noBG_mIoU,
                                                    (sum(noBG_correct_class) / float(sum(noBG_seen_class))),(np.mean(np.array(noBG_correct_class) / (np.array(noBG_seen_class, dtype=np.float64) + 1e-6))))
        io.cprint(outstr_without_background)

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASS):
            iou_per_class_str += 'class %s percentage: %.4f, IoU: %.4f \n' % (
                labels2categories[l] + ' ' * (16 - len(labels2categories[l])), labelweights[l],
                total_correct_class[l] / float(total_iou_deno_class[l]))
        io.cprint(iou_per_class_str)
        io.cprint('\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--model', type=str, default='dgcnn_patch', metavar='N',
                        choices=['dgcnn','dgcnn_patch'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--data_dir', type=str, default='/home/bi/study/thesis/data/test_6', 
                        help='file need to be tested')
    parser.add_argument('--finetune', type=int, default=1, metavar='N',
                        help='if we finetune the model')
    parser.add_argument('--num_segmentation_type', type=int, default=6, metavar='num_segmentation_type',
                        help='num_segmentation_type)')
    parser.add_argument('--test', type=int, default=1,
                        help='evaluate the model')
    parser.add_argument('--training', type=int, default=0,
                        help='evaluate the model')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--use_class_weight', type=bool, default=False,
                        help='enables using weights')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--vis', type=int, default=1, metavar='S',
                        help='if visualize the predicted result')
    parser.add_argument('--save', type=int, default=1, metavar='S',
                        help='if save the predicted result')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--screw_weight', type=float, default=1,
                        help='screw_weight')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--npoints', type=int, default=2048, 
                        help='Point Number [default: 4096]')
    parser.add_argument('--test_symbol', type=str, default='Test', 
                        help='Which datablocks to use for test')
    parser.add_argument('--num_heads', type=int, default=4, metavar='num_attention_heads',
                        help='number of attention_heads for self_attention ')
    parser.add_argument('--num_attention_layer', type=int, default=1, metavar='num_attention_hea',
                        help='number of attention_heads for self_attention ')
    parser.add_argument('--self_encoder_latent_features', type=int, default=128, metavar='self_encoder_latent_features',
                        help='number of hidden size for self_attention ')
    parser.add_argument('--hidden_size_for_cross_attention', type=int, default=512, metavar='hidden_size_for_cross_attention',
                        help='number of hidden size for cross attention')
    parser.add_argument('--after_stn_as_input', type=int, default=1, metavar='S',
                        help='get the neighbors from rotated points for each superpoints')
    parser.add_argument('--after_stn_as_kernel_neighbor_query', type=int, default=1, metavar='S',
                        help='get the high dimensional features for patch sample to get super points with rotated points')
    args = parser.parse_args()

    NUM_CLASS=args.num_segmentation_type
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    test(args)
