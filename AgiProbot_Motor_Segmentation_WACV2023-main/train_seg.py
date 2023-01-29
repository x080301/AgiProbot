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
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dataloader import MotorDataset,MotorDataset_validation
from model import DGCNN_semseg ,DGCNN_patch_semseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss,mean_loss,normalize_data,feature_transform_reguliarzer,rotate_per_batch,PrintLog
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes = ['clamping_system', 'cover', 'gear_container', 'charger', 'bottom', 'side_screw','cover_screw']                                                                                                                               
labels2categories={i:cls for i,cls in enumerate(classes)}       #dictionary for labels2categories



def _init_(add_string,exp_name):
    if not os.path.exists('outputs'):       #initial the file, if not exiting (os.path.exists() is pointed at ralative position and current cwd)
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.model+'/'+args.which_dataset):
        os.makedirs('outputs/'+args.model+'/'+args.which_dataset)
    if not os.path.exists('outputs/'+args.model+'/'+args.which_dataset+'/'+exp_name+add_string+'/'+'models'):
        os.makedirs('outputs/'+args.model+'/'+args.which_dataset+'/'+exp_name+add_string+'/'+'models') 


          
def train(args, io):
    NUM_POINT=args.npoints
    print("start loading training data ...")
    TRAIN_DATASET = MotorDataset(split='train', data_root=args.data_dir, num_points=NUM_POINT,screw_weight=args.screw_weight, test_area=args.validation_symbol, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = MotorDataset_validation(split='test', data_root=args.data_dir, num_points=NUM_POINT, test_area=args.validation_symbol, sample_rate=1.0, transform=None)
    train_loader = DataLoader(TRAIN_DATASET, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True,worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    test_loader = DataLoader(TEST_DATASET, num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    io.cprint(args)
    #Try to load models
    tmp=torch.cuda.max_memory_allocated()
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    elif args.model == 'dgcnn_patch':
        model = DGCNN_patch_semseg(args).to(device)
    else:
        raise Exception("Not implemented")
    #summary(model,input_size=(3,4096),batch_size=1,device='cuda')

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.1, args.epochs)
    
    ## if finetune is true, the the best_finetune will be cosidered first, then best.pth will be taken into consideration
    if args.finetune:
        if os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+"/models/best_finetune.pth"):
            checkpoint = torch.load(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+"/models/best_finetune.pth")
            print('Use pretrain finetune model to finetune')
            
        elif os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+"/models/best.pth"):
            checkpoint = torch.load(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+"/models/best.pth")
            print('Use pretrain model to finetune')
        else:
            print('no exiting pretrained model to finetune')
            exit(-1)
        if not os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+"/models/best_finetune.pth") and os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+"/models/best.pth"):
            start_epoch = 0
        else:
            start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        
    else:
        if os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+"/models/best.pth"):
            checkpoint = torch.load(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+"/models/best.pth")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Use pretrain model')
        else:
            start_epoch=0
            print('no exiting pretrained model,starting from scratch')



    criterion = cal_loss
    criterion2= mean_loss
    best_iou = 0
    best_screws_iou=0
    weights__ = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    persentige = torch.Tensor(TRAIN_DATASET.persentage).cuda()
    io.cprint(persentige)
    scale=weights__*persentige
    scale=1/scale.sum()
    
    # weights__*=scale
    if not args.use_class_weight:
        for i in range(6):
            weights__[i]=1

    for epoch in range(start_epoch,args.epochs):
        ####################
        # Train
        ####################
        num_batches=len(train_loader)
        total_correct=0
        total_seen=0
        loss_sum=0
        model=model.train()
        args.training=True
        total_correct_class__ = [0 for _ in range(NUM_CLASS)]
        total_iou_deno_class__ = [0 for _ in range(NUM_CLASS)]
        for i,(points,target,goals,masks) in tqdm(enumerate(train_loader),total=len(train_loader),smoothing=0.9):
            points, target,goals,masks = points.to(device), target.to(device) ,goals.to(device) ,masks.to(device)       #(batch_size, num_points, features)    (batch_size, num_points)
            points=normalize_data(points) 
            if args.after_stn_as_kernel_neighbor_query:                              #[bs,4096,3]
                points,goals,GT=rotate_per_batch(points,goals)
            else:
                points,_,GT=rotate_per_batch(points,goals)
            points = points.permute(0, 2, 1)                            #(batch_size,features,numpoints)
            batch_size = points.size()[0]
            opt.zero_grad()
            seg_pred,trans,result = model(points.float(),target)        #(batch_size, class_categories, num_points)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()                      #(batch_size,num_points, class_categories)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()              #array(batch_size*num_points)            loss = criterion(seg_pred.view(-1, NUM_CLASS), target.view(-1,1).squeeze(),weights,using_weight=args.use_weigth)     #a scalar
            loss = criterion(seg_pred.view(-1, NUM_CLASS), target.view(-1,1).squeeze(),weights__,using_weight=args.use_class_weight)     #a scalar
            if args.model=="PCT_patch" or args.model=="dgcnn_patch":
                loss=loss+criterion2(result.view(-1, 3),goals.view(-1, 3),masks)*args.kernel_loss_weight
            loss = loss+feature_transform_reguliarzer(trans)*args.stn_loss_weight
            loss.backward()
            opt.step()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASS)                    # (batch_size*num_points , num_class)
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()                     # array(batch_size*num_points)
            correct = np.sum(pred_choice == batch_label)                            # when a np arraies have same shape, a==b means in conrrespondending position it equals to one,when they are identical
            total_correct += correct
            total_seen += (batch_size * NUM_POINT)
            loss_sum += loss
            for l in range(NUM_CLASS):
                total_correct_class__[l] += np.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class__[l] += np.sum(((pred_choice == l) | (batch_label == l)))
        mIoU__ = np.mean(np.array(total_correct_class__) / (np.array(total_iou_deno_class__, dtype=np.float64) + 1e-6))

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        outstr = 'Train %d, loss: %.6f, train acc: %.6f ' % (epoch,(loss_sum / num_batches),(total_correct / float(total_seen)))
        io.cprint(outstr)
        writer.add_scalar('Training loss', (loss_sum / num_batches), epoch)
        writer.add_scalar('Training accuracy', (total_correct / float(total_seen)), epoch)
        writer.add_scalar('Training mean ioU', (mIoU__), epoch)
        writer.add_scalar('Training loU of screw',(total_correct_class__[5] / float(total_iou_deno_class__[5])), epoch)
        ####################
        # Validation
        ####################
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
            args.training=False

            for i,(points,seg) in tqdm(enumerate(test_loader),total=len(test_loader),smoothing=0.9):
                points, seg = points.to(device), seg.to(device)
                points=normalize_data(points)
                points,GT=rotate_per_batch(points,None)
                points = points.permute(0, 2, 1)
                batch_size = points.size()[0]
                seg_pred_,trans_,_= model(points.float(),seg)
                seg_pred_ = seg_pred_.permute(0, 2, 1).contiguous()
                batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()  
                loss = criterion(seg_pred_.view(-1, NUM_CLASS), seg.view(-1,1).squeeze(),weights__,using_weight=args.use_class_weight)     #a scalar
                loss = loss+feature_transform_reguliarzer(trans_)*args.stn_loss_weight
                seg_pred_ = seg_pred_.contiguous().view(-1, NUM_CLASS)   
                pred_choice = seg_pred_.cpu().data.max(1)[1].numpy()  
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (batch_size * NUM_POINT)
                loss_sum+=loss
                tmp, _ = np.histogram(batch_label, range(NUM_CLASS + 1))
                labelweights += tmp
                for l in range(NUM_CLASS):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))


               ####### calculate without Background ##############
                for l in range(1, NUM_CLASS):
                    noBG_seen_class[l-1] += np.sum((batch_label == l))
                    noBG_correct_class[l-1] += np.sum((pred_choice == l) & (batch_label == l))
                    noBG_iou_deno_class[l-1] += np.sum(((pred_choice == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))

            outstr = 'Validation with backgroud----epoch: %d,  eval loss %.6f,  eval mIoU %.6f,  eval point acc %.6f, eval avg class acc %.6f' % (epoch,(loss_sum / num_batches),mIoU,
                                                        (total_correct / float(total_seen)),(np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))

            io.cprint(outstr)
            noBG_mIoU = np.mean(np.array(noBG_correct_class) / (np.array(noBG_iou_deno_class, dtype=np.float64) + 1e-6))
            outstr_without_background='Validation without backgroud----epoch: %d, mIoU %.6f,  eval point acc: %.6f, eval avg class acc: %.6f' % (epoch,noBG_mIoU,
                                                        (sum(noBG_correct_class) / float(sum(noBG_seen_class))),(np.mean(np.array(noBG_correct_class) / (np.array(noBG_seen_class, dtype=np.float64) + 1e-6))))
            io.cprint(outstr_without_background)

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASS):
                iou_per_class_str += 'class %s percentage: %.4f, loss_weight: %.4f, IoU: %.4f \n' % (
                    labels2categories[l] + ' ' * (16 - len(labels2categories[l])), labelweights[l],weights__[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))
            io.cprint(iou_per_class_str)

            if mIoU >= best_iou:
                best_iou = mIoU
                if args.finetune:
                    savepath = str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+"/models/best_finetune_m.pth"
                else:
                    savepath = str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+"/models/best_m.pth"

                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }
                io.cprint('Saving best model at %s' % savepath)
                torch.save(state, savepath)
            io.cprint('Best mIoU: %f' % best_iou)
            cur_screws_iou=total_correct_class[5] / float(total_iou_deno_class[5])
            if cur_screws_iou >= best_screws_iou:
                best_screws_iou=cur_screws_iou
                if args.finetune:
                    savepath = str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+"/models/best_finetune.pth"
                else:
                    savepath = str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+"/models/best.pth"
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }
                io.cprint('Saving best model at %s' % savepath)
                torch.save(state, savepath)
            io.cprint('Best IoU of screw: %f' % best_screws_iou)
            io.cprint('\n\n')
        writer.add_scalar('learning rate',opt.param_groups[0]['lr'],  epoch)
        writer.add_scalar('Validation loss', (loss_sum / num_batches), epoch)
        writer.add_scalar('Validation accuracy', (total_correct / float(total_seen)), epoch)
        writer.add_scalar('validation mean ioU', (mIoU), epoch)
        writer.add_scalar('Validation loU of screw',(total_correct_class[5] / float(total_iou_deno_class[5])), epoch)

    io.close()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--model', type=str, default='dgcnn_patch', metavar='N',
                        choices=['dgcnn','dgcnn_patch'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--batch_size', type=int, default=3, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--data_dir', type=str, default='/home/bi/study/thesis/data/training_6', 
                        help='file need to be tested')
    parser.add_argument('--which_dataset', type=str, default='Motor_Dataset4', metavar='N',
                        help='experiment version to record reslut')
    parser.add_argument('--exp_name', type=str, default='STN_16_2048_100', metavar='N',
                        help='experiment version to record reslut')
    parser.add_argument('--training',  type=int, default=1,
                        help='evaluate the model')  
    parser.add_argument('--finetune', type=int, default=0,
                        help='if we finetune the model')
    parser.add_argument('--test', type=int, default=0,
                        help='evaluate the model')
    parser.add_argument('--num_segmentation_type', type=int, default=6, metavar='num_segmentation_type',
                        help='num_segmentation_type)')
    parser.add_argument('--kernel_loss_weight', type=float, default=0.05, metavar='F',
                        help='factor of loss_cluster')
    parser.add_argument('--stn_loss_weight', type=float, default=0.01, metavar='F',
                        help='factor of loss_cluster')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_class_weight', type=int, default=0,
                        help='enables using weights')
    parser.add_argument('--screw_weight', type=float, default=1.0, metavar='F',
                        help='weight for screw')
    parser.add_argument('--use_sgd', type=int, default=0,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=int, default=0,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--npoints', type=int, default=2048, 
                        help='Point Number [default: 4096]')
    parser.add_argument('--validation_symbol', type=str, default='Validation', 
                        help='Which datablocks to use for validation')
    parser.add_argument('--test_symbol', type=str, default='Train', 
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

    if args.finetune==True:
        add_string='_finetune'
    else:
        add_string=''
    _init_(add_string,args.exp_name)


    if not os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/0'):
        os.makedirs(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/0')
        path=str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/0'
        writer = SummaryWriter(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/0')
        io = PrintLog(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/0'+'/run'+add_string+'.log')
    else:
        i=1
        while(True):
            if not os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/'+str(i)):
                os.makedirs(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/'+str(i))
                path=str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/'+str(i)
                writer = SummaryWriter(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/'+str(i))
                io = PrintLog(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/'+str(i)+'/run'+add_string+'.log')
                break
            else:
                i+=1
        i=0
        path=str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/'+str(i)
        while os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/'+str(i)):
            path=str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/'+str(i)
            io_test = PrintLog(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.which_dataset+'/'+args.exp_name+add_string+'/'+str(i)+'/Test'+add_string+'.log')
            i+=1 

    os.system('cp train_semseg.py '+path+'/'+'train_semseg.py.backup')
    os.system('cp util.py '+path+'/'+'util.py.backup')
    os.system('cp model_rotation.py '+path+'/'+'model_rotation.py.backup')
    os.system('cp dataloader.py '+path+'/'+'dataloader.py.backup')
    os.system('cp train.sh '+path+'/'+'train.sh.backup')

    NUM_CLASS=args.num_segmentation_type
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    train(args, io)
