#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: bixuelei
@Contact: xueleibi@gmail.com
@File: model.py
@Time: 2022/1/15 17:11 PM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import create_conv1d_serials,create_conv3d_serials,init_weights,index_points,square_distance,get_neighbors,SA_Layer_Multi_Head,PTransformerDecoderLayer,PTransformerDecoder,SA_Layers
from torch.autograd import Variable



class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) #bs features 2048
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x



class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k
        self.s3n=STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn8,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 6, kernel_size=1, bias=False)   #256*6=1536
        #dgcnn_con      1244800

        

    def forward(self, x,input_for_alignment_all_structure):
        batch_size = x.size(0)
        num_points = x.size(2)
        x=x.float() 

        trans=self.s3n(x)
        x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        x=x.permute(0,2,1)

        x = get_neighbors(x, k=self.k)         # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
        return x,trans,None



class ball_query_sample_with_goal(nn.Module):                                #top1(ball) to approach 
    def __init__(self,args, num_feats, input_dims, actv_fn=F.relu, top_k=16):
        """This function returns a sorted Tensor of Points. The Points are sorted

        """
        super(ball_query_sample_with_goal, self).__init__()
        self.point_after=args.after_stn_as_kernel_neighbor_query
        self.args=args
        self.num_heads=args.num_heads
        self.num_layers=args.num_attention_layer
        self.num_latent_feats_inencoder=args.self_encoder_latent_features
        self.num_feats = num_feats
        self.actv_fn = actv_fn
        self.input_dims = input_dims

        self.top_k = 32
        self.d_model = 480
        self.radius = 0.3
        self.max_radius_points = 32

        self.self_atn_layer =SA_Layer_Multi_Head(args,256)
        self.selfatn_layers=SA_Layers(self.num_layers,self.self_atn_layer)

        self.loss_function=nn.MSELoss()
 
        self.feat_channels_1d = [self.num_feats,64, 64, 48]
        self.feat_generator = create_conv1d_serials(self.feat_channels_1d)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.feat_channels_1d[i+1])
                for i in range(len(self.feat_channels_1d)-1)
            ]
        )


        self.feat_channels_3d = [128, 256, self.d_model]
        self.radius_cnn = create_conv3d_serials(self.feat_channels_3d, self.max_radius_points, 3)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList(
            [
                nn.BatchNorm3d(num_features=self.feat_channels_3d[i])
                for i in range(len(self.feat_channels_3d))
            ]
        )

    def forward(self, hoch_features, input,x_a_r,target):                                          #[bs,1,features,n_points]   [bs,C,n_points]

        top_k = self.top_k                                                             
        origial_hoch_features = hoch_features                                          #[bs,features,n_points]
        feat_dim = input.shape[1]                                                      

        hoch_features_att = hoch_features 
        #############################################################
        #implemented by myself
        #############################################################
        hoch_features_att=hoch_features_att.permute(0,2,1)                          #[bs,features,n_points]->#[bs,n_points,features]
        hoch_features_att=self.selfatn_layers(hoch_features_att)                    #[bs,n_points,features]->#[bs,n_points,features]
        hoch_features_att=hoch_features_att.permute(0,2,1)                          #[bs,n_points,features]->#[bs,features,n_points] 


        ##########################
        #
        ##########################
        high_inter=hoch_features_att
        for j, conv in enumerate(self.feat_generator):                      #[bs,features,n_points]->[bs,48,n_points]
            bn = self.feat_bn[j]
            high_inter = self.actv_fn(bn(conv(high_inter)))
        topk = torch.topk(high_inter, k=top_k, dim=-1)                      #[bs,48,n_points]->[bs,48,32]  48 is number of patch,32 the numbet of points for each patch
        indices_32 = topk.indices                                          #[bs,48,32]
        indices=indices_32[:,:,0]                                           #[bs,48,1]  center points for each patch
        result_net =torch.ones((1))
        if not self.args.test and self.args.training:                       #index the centered point for each patch        
            result_net = index_points(input.permute(0, 2, 1).float(), indices)



        sorted_input = torch.zeros((origial_hoch_features.shape[0], feat_dim, top_k)).to(    
            input.device                                                        #[bs,C,n_superpoint]
        )

        if top_k == 1:
            indices = indices.unsqueeze(dim=-1)

        sorted_input = index_points(input.permute(0, 2, 1).float(), indices).permute(0, 2, 1)       #[bs,n_superpoint]->[bs,C,n_superpoint]

        all_points = input.permute(0, 2, 1).float()                              #[bs,C,n_points]->[bs,n_points,C]
        query_points = sorted_input.permute(0, 2, 1)                             #[bs,C,n_superpoint]->[bs,n_superpoint,C]

        dis1=square_distance(all_points,query_points)                               
        radius_indices=torch.topk(-dis1, k=32, dim=-2).indices.permute(0, 2, 1)  


        if self.point_after:
            radius_points = index_points(x_a_r.permute(0,2,1), radius_indices)
        else:
            radius_points = index_points(all_points, radius_indices)                    #[bs,n_superpoint,n_sample,C]
        radius_points=radius_points.unsqueeze(dim=1)                                  

        for i, radius_conv in enumerate(self.radius_cnn):                           #[bs,n_superpoint,n_sample+1,C]->[bs,512,n_superpoint,1,1]
            bn = self.radius_bn[i]
            radius_points = self.actv_fn(bn(radius_conv(radius_points)))

        radius_points = radius_points.squeeze(dim=-1).squeeze(dim=-1)             #[bs,512,n_superpoint,1,1]->[bs,512,n_superpoint]                                          #[bs,n_superpoint]
        radius_points = torch.cat((radius_points, indices_32.permute(0, 2, 1)), dim=1)
        return radius_points,result_net



class DGCNN_patch_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_patch_semseg, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.source_sample_after_rotate=args.after_stn_as_input
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 32
        self.d_model = args.hidden_size_for_cross_attention
        self.num_classes= 6  

        #########################################################################
        # dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(512)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 512, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
     




        #############################################################################
        # self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)   
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )
        self.superpointnet =ball_query_sample_with_goal(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)

        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=512)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 1, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=1,num_decoder_layers=1,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        #final segmentation layer
        self.bn7 = nn.BatchNorm1d(1024)
        self.conv7 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),         
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn8 = nn.BatchNorm1d(512)
        self.conv8 = nn.Sequential(nn.Conv1d(1728, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn9 = nn.BatchNorm1d(256)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn9,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)
        self.conv10 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x, target):
        x=x.float()
        x_local=x
        input=x

        trans=self.s3n(x)
        x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        #Visuell_PointCloud_per_batch(x,target)
        x=x.permute(0,2,1)
        x_a_r=x
        #############################################
        ## Global Features
        #############################################
        batch_size = x.size(0)
        num_points = x.size(2)
 
        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x_pre = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x_mid = self.conv6(x_pre)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)


        #############################################
        ## sample Features
        #############################################  
        if self.source_sample_after_rotate:
            x_sample=x_a_r  
        else:
            x_sample=input  

        for i, sort_conv in enumerate(self.sort_cnn):                       #[bs,1,Cor,n_points]->[bs,256,n_points]

            bn = self.sort_bn[i]
            x_sample = self.actv_fn(bn(sort_conv(x_sample)))


        x_patch, predicted_kernels = self.superpointnet(x_sample,input,x_a_r,target)              #[bs,256,n_points]->[bs,512,16]


        #############################################
        ## Point Transformer  patch to global
        #############################################
        source = x_patch.permute(2, 0, 1)                            # [bs,512,64]->[64,bs,1024]
        target = x_mid.permute(2, 0, 1)                                 # [bs,1024,10]->[10,bs,512]
        embedding = self.transformer_model(source, target)                 # [64,bs,1024]+[16,bs,1024]->[16,bs,1024]


        # #############################################
        # ## Point Transformer  local to global
        # #############################################
        # target = x.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        # source = x_patch.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        # embedding = self.transformer_model(source, target)                 # [16,bs,1024]+[64,bs,1024]->[64,bs,1024]



        ################################################
        ##segmentation
        ################################################
        embedding=embedding.permute(1,2,0)                                  # [32,bs,512]->[bs,512,32]
        # x=self.bn7(self.conv7(x))
        x = self.conv7(x_mid)
        x = x.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_pre,embedding), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv8(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        # x = self.dp1(x)
        x = self.conv9(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)
        if self.args.training:
            return x,trans,predicted_kernels
        else:
            return x,trans,None

