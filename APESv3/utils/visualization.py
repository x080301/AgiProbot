import os
import shutil
import numpy as np
import pkbar
import math
from plyfile import PlyData, PlyElement
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.image as Image
from matplotlib import cm
from collections import OrderedDict
from .visualization_data_processing import *


def visualize_shapenet_predictions(config, samples, preds, seg_labels, cls_label, shape_ious, index, artifacts_path):
    base_path = f'{artifacts_path}/vis_pred'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    category_id_to_hash_code_mapping = {}
    for hash_code in list(config.datasets.mapping.keys()):
        category_id_to_hash_code_mapping[str(config.datasets.mapping[hash_code]['category_id'])] = hash_code
    categories = []
    for cat_id in cls_label:
        hash_code = category_id_to_hash_code_mapping[str(cat_id)]
        categories.append(config.datasets.mapping[hash_code]['category'])
    # select predictions
    samples_tmp = []
    preds_tmp = []
    seg_gts_tmp = []
    categories_tmp = []
    ious_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.downsample.M))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_preds.vis_which:
        samples_toappend = samples[cls_label == cat_id][:config.test.visualize_preds.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        preds_tmp.append(preds[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        seg_gts_tmp.append(seg_labels[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        categories_tmp.append(np.asarray(categories)[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        ious_tmp.append(np.asarray(shape_ious)[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_label == cat_id][:config.test.visualize_preds.num_vis])
    samples = np.concatenate(samples_tmp)
    preds = np.concatenate(preds_tmp)
    seg_labels = np.concatenate(seg_gts_tmp)
    categories = np.concatenate(categories_tmp)
    shape_ious = np.concatenate(ious_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized prediction files, please wait...', target=len(samples))
    i_saved = 0
    if config.datasets.dataset_name == "shapenet_AnTao350M":
        view_range = 0.6
    elif config.datasets.dataset_name == "shapenet_Yi650M" or config.datasets.dataset_name == "shapenet_Normal":
        view_range = 0.3
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    for i, (sample, pred, seg_gt, category, iou) in enumerate(zip(samples, preds, seg_labels, categories, shape_ious)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        xyzRGB_gt = []
        xyzRGB_list = []
        xyzRGB_gt_list = []
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for xyz, p, gt in zip(sample, pred, seg_gt):
            xyzRGB_tmp = []
            xyzRGB_gt_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend(config.datasets.cmap[str(p)])
            xyzRGB.append(tuple(xyzRGB_tmp))
            xyzRGB_gt_tmp.extend(list(xyz))
            xyzRGB_gt_tmp.extend(config.datasets.cmap[str(gt)])
            xyzRGB_gt.append(tuple(xyzRGB_gt_tmp))
        xyzRGB_list.append(xyzRGB)
        xyzRGB_gt_list.append(xyzRGB_gt)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            xyzRGB = []
            xyzRGB_gt = []
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer + 1):
                    idx = index[layer - j][i, 0][idx]  # index mapping
            else:
                idx = index[layer][i, 0]
            for xyz, p, gt in zip(sample[idx], pred[idx], seg_gt[idx]):
                xyzRGB_tmp = []
                xyzRGB_gt_tmp = []
                xyzRGB_tmp.extend(list(xyz))
                xyzRGB_tmp.extend(config.datasets.cmap[str(p)])
                xyzRGB.append(tuple(xyzRGB_tmp))
                xyzRGB_gt_tmp.extend(list(xyz))
                xyzRGB_gt_tmp.extend(config.datasets.cmap[str(gt)])
                xyzRGB_gt.append(tuple(xyzRGB_gt_tmp))
            xyzRGB_list.append(xyzRGB)
            xyzRGB_gt_list.append(xyzRGB_gt)
        if config.test.visualize_preds.format == 'ply':
            for which_layer, (xyzRGB, xyzRGB_gt) in enumerate(zip(xyzRGB_list, xyzRGB_gt_list)):
                if which_layer > 0:
                    pred_saved_path = f'{cat_path}/{category}{i-i_saved}_pred_{math.floor(iou * 1e5)}_dsLayer{which_layer}.ply'
                    gt_saved_path = f'{cat_path}/{category}{i-i_saved}_gt_dsLayer{which_layer}.ply'
                else:
                    pred_saved_path = f'{cat_path}/{category}{i-i_saved}_pred_{math.floor(iou * 1e5)}.ply'
                    gt_saved_path = f'{cat_path}/{category}{i-i_saved}_gt.ply'
                vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(pred_saved_path)
                vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(gt_saved_path)
        elif config.test.visualize_preds.format == 'png':
            for which_layer, (xyzRGB, xyzRGB_gt) in enumerate(zip(xyzRGB_list, xyzRGB_gt_list)):
                if which_layer > 0:
                    pred_saved_path = f'{cat_path}/{category}{i-i_saved}_pred_{math.floor(iou * 1e5)}_dsLayer{which_layer}.png'
                    gt_saved_path = f'{cat_path}/{category}{i-i_saved}_gt_dsLayer{which_layer}.png'
                else:
                    pred_saved_path = f'{cat_path}/{category}{i-i_saved}_pred_{math.floor(iou * 1e5)}.png'
                    gt_saved_path = f'{cat_path}/{category}{i-i_saved}_gt.png'
                vertex = np.array(xyzRGB)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-view_range, view_range)
                ax.set_ylim3d(-view_range, view_range)
                ax.set_zlim3d(-view_range, view_range)
                ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(pred_saved_path, bbox_inches='tight')
                plt.close(fig)
                vertex = np.array(xyzRGB_gt)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-view_range, view_range)
                ax.set_ylim3d(-view_range, view_range)
                ax.set_zlim3d(-view_range, view_range)
                ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(gt_saved_path, bbox_inches='tight')
                plt.close(fig)
        else:
            raise ValueError(f'format should be png or ply, but got {config.test.visualize_preds.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_shapenet_downsampled_points(config, samples, index, cls_label, shape_ious, artifacts_path):
    base_path = f'{artifacts_path}/vis_ds_points'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    category_id_to_hash_code_mapping = {}
    for hash_code in list(config.datasets.mapping.keys()):
        category_id_to_hash_code_mapping[str(config.datasets.mapping[hash_code]['category_id'])] = hash_code
    categories = []
    for cat_id in cls_label:
        hash_code = category_id_to_hash_code_mapping[str(cat_id)]
        categories.append(config.datasets.mapping[hash_code]['category'])
    # select samples
    samples_tmp = []
    categories_tmp = []
    ious_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.downsample.M))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.downsample.M))]
    i_categories =[]
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(np.asarray(categories)[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        ious_tmp.append(np.asarray(shape_ious)[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    shape_ious = np.concatenate(ious_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    if config.datasets.dataset_name == "shapenet_AnTao350M":
        view_range = 0.6
    elif config.datasets.dataset_name == "shapenet_Yi650M" or config.datasets.dataset_name == "shapenet_Normal":
        view_range = 0.3
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    for i, (sample, category, iou) in enumerate(zip(samples, categories, shape_ious)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer+1):
                    idx = index[layer-j][i, 0][idx]   # index mapping
            else:
                idx = index[layer][i, 0]
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i-i_saved}_layer{layer}_{math.floor(iou * 1e5)}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i-i_saved}_layer{layer}_{math.floor(iou * 1e5)}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-view_range, view_range)
                ax.set_ylim3d(-view_range, view_range)
                ax.set_zlim3d(-view_range, view_range)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o', s=1)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_downsampled_points(config, samples, index, cls_labels, mode, artifacts_path):
    base_path = f'{artifacts_path}/vis_ds_points/{mode}'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.downsample.M))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.downsample.M))]
    i_categories =[]
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    for i, (sample, category) in enumerate(zip(samples, categories)):
        
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category

        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer+1):
                    idx = index[layer-j][i, 0][idx]   # index mapping
            else:
                idx = index[layer][i, 0]
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i-i_saved}_layer{layer}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i-i_saved}_layer{layer}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-0.6, 0.6)
                ax.set_ylim3d(-0.6, 0.6)
                ax.set_zlim3d(-0.6, 0.6)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o', s=1)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:]/255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_heatmap(config, samples, attention_map, cls_labels, artifacts_path):
    # this function only generates heatmap for the first downsample layer
    my_cmap = cm.get_cmap('viridis_r', samples.shape[1])
    base_path = f'{artifacts_path}/vis_heatmap/one'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.edgeconv_block.enable:
        attention_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.downsample.M))]
    if config.neighbor2point_block.enable:
        attention_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    if config.point2point_block.enable:
        attention_tmp = [[] for _ in range(len(config.point2point_block.downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_attention_heatmap.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
        for layer, atten in enumerate(attention_map):
            attention_tmp[layer].append(atten[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    attention_map = []
    for each in attention_tmp:
        attention_map.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized heatmap files, please wait...', target=len(samples))
    
    i_saved = 0
    for i, (sample, category, atten) in enumerate(zip(samples, categories, attention_map[0])):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        atten = atten[0]
        atten = (atten - np.mean(atten)) / np.std(atten) + 0.5
        for xyz, rgb in zip(sample, atten):
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            RGB = 255 * np.asarray(my_cmap(rgb))[:3]
            xyzRGB_tmp.extend(list(RGB))
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        if config.test.visualize_attention_heatmap.format == 'ply':
            saved_path = f'{cat_path}/{category}{i-i_saved}.ply'
            vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
            PlyData([vertex]).write(saved_path)
        elif config.test.visualize_attention_heatmap.format == 'png':
            saved_path = f'{cat_path}/{category}{i-i_saved}.png'
            vertex = np.array(xyzRGB)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-0.6, 0.6)
            ax.set_ylim3d(-0.6, 0.6)
            ax.set_zlim3d(-0.6, 0.6)
            ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:]/255, marker='o', s=1)
            plt.axis('off')
            plt.grid('off')
            plt.savefig(saved_path, bbox_inches='tight')
            plt.close(fig)
        else:
            raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_downsampled_points_rs_fps(config, samples, index, cls_labels, ds_type):
    base_path = f'./artifacts/{config.wandb.name}/vis_ds_points/{ds_type}'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.downsample.M))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(3)]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.downsample.M))]
    i_categories =[]
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    for i, (sample, category) in enumerate(zip(samples, categories)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            idx = index[layer][i, 0]
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i-i_saved}_layer{layer}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i-i_saved}_layer{layer}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-0.6, 0.6)
                ax.set_ylim3d(-0.6, 0.6)
                ax.set_zlim3d(-0.6, 0.6)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o', s=1)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:]/255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_heatmap_mode(config, samples, attention_map, cls_labels, mode, artifacts_path):
    # this function only generates heatmap for the first downsample layer
    my_cmap = cm.get_cmap('viridis_r', samples.shape[1])
    base_path = f'{artifacts_path}/vis_heatmap/{mode}'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.neighbor2point_block.enable:
        attention_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_attention_heatmap.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
        for layer, atten in enumerate(attention_map):
            attention_tmp[layer].append(atten[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    attention_map = []
    for each in attention_tmp:
        attention_map.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized heatmap files, please wait...', target=len(samples))
    
    i_saved = 0
    for i, (sample, category, atten) in enumerate(zip(samples, categories, attention_map[0])):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        atten = atten[0]
        if mode == "trained_boltzmann":
            atten = np.log(atten)
            atten = atten - np.mean(atten) + 0.5
        else:
            atten = (atten - np.mean(atten)) / np.std(atten) + 0.5
        for xyz, rgb in zip(sample, atten):
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            RGB = 255 * np.asarray(my_cmap(rgb))[:3]
            xyzRGB_tmp.extend(list(RGB))
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        if config.test.visualize_attention_heatmap.format == 'ply':
            saved_path = f'{cat_path}/{category}{i-i_saved}.ply'
            vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
            PlyData([vertex]).write(saved_path)
        elif config.test.visualize_attention_heatmap.format == 'png':
            saved_path = f'{cat_path}/{category}{i-i_saved}.png'
            vertex = np.array(xyzRGB)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-0.6, 0.6)
            ax.set_ylim3d(-0.6, 0.6)
            ax.set_zlim3d(-0.6, 0.6)
            ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:]/255, marker='o', s=1)
            plt.axis('off')
            plt.grid('off')
            plt.savefig(saved_path, bbox_inches='tight')
            plt.close(fig)
        else:
            raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')

def visualize_modelnet_heatmap_compare(config, samples, heatmap_dict, cls_labels, artifacts_path):
    # this function only generates heatmap for the first downsample layer
    my_cmap = cm.get_cmap('viridis_r', samples.shape[1])
    base_path = f'{artifacts_path}/vis_heatmap/compare'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    attention_tmp_dict = {}
    if config.neighbor2point_block.enable:
        for mode in heatmap_dict.keys():
            attention_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
            attention_tmp_dict[mode] = attention_tmp
    i_categories = []
    for cat_id in config.test.visualize_attention_heatmap.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
        for mode, map in heatmap_dict.items():
            for layer, atten in enumerate(map):
                attention_tmp_dict[mode][layer].append(atten[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    attention_map_dict = OrderedDict()
    for mode, map in attention_tmp_dict.items():
        attention_map_dict[mode] = []
        for each in map:
            attention_map_dict[mode].append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized heatmap files, please wait...', target=len(samples))

    if not config.test.visualize_attention_heatmap.format == 'png':
        raise ValueError(f'format should be png, but got {config.test.visualize_downsampled_points.format}')
    
    i_saved = 0
    for i, (sample, category) in enumerate(zip(samples, categories)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        saved_path = f'{cat_path}/{category}{i-i_saved}.png'        
        fig = plt.figure()
        for i_plt, (mode, map_mode) in enumerate(attention_map_dict.items()):
            atten = map_mode[0][i]
            xyzRGB = []
            atten = atten[0]
            atten = (atten - np.mean(atten)) / np.std(atten) + 0.5 # normalization
            for xyz, rgb in zip(sample, atten):
                xyzRGB_tmp = []
                xyzRGB_tmp.extend(list(xyz))
                RGB = 255 * np.asarray(my_cmap(rgb))[:3]
                xyzRGB_tmp.extend(list(RGB))
                xyzRGB.append(xyzRGB_tmp)
            vertex = np.array(xyzRGB)
            ax = fig.add_subplot(2, 3, i_plt + 1, projection='3d')
            ax.set_xlim3d(-0.6, 0.6)
            ax.set_ylim3d(-0.6, 0.6)
            ax.set_zlim3d(-0.6, 0.6)
            sc = ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:]/255, marker='o', s=1)
            ax.set_title(f'{mode}')
            plt.axis('off')
            plt.grid('off')
        plt.tight_layout()

        # cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # 调整位置和大小以适合图形
        # cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
        # cbar.set_label('Color Scale')
        # plt.tight_layout()
        plt.savefig(saved_path, bbox_inches='tight')
        plt.close(fig)
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')
    

def visualize_modelnet_combine(config, artifacts_path):
    base_path = artifacts_path
    ds_points_path = os.path.join(base_path, 'vis_ds_points')
    heatmap_path = os.path.join(base_path, 'vis_heatmap')
    to_combine_paths = {
        'heatmap': heatmap_path,
        'ds_points': ds_points_path
    }
    save_base_path = os.path.join(base_path, 'vis_compare')
    
    attention_modes = os.listdir(ds_points_path)
    attention_modes.sort()
    categories = os.listdir(os.path.join(ds_points_path, attention_modes[0]))
    categories.sort()
    
    categories_dict = {}
    num_images = 0
    for category in categories:
        cat_hm_images = os.listdir(os.path.join(heatmap_path, attention_modes[0], category))
        num_images += len(cat_hm_images)
        categories_dict[category] = cat_hm_images
    
    pbar = pkbar.Pbar(name='Generating visualized combined heatmap and downsampled points files, please wait...', target=num_images)
    
    i_pbar = 0
    for i, (cat, hm_imgs) in enumerate(categories_dict.items()):
        save_cat_path = os.path.join(save_base_path, cat)
        if not os.path.exists(save_cat_path):
            os.makedirs(save_cat_path)
        
        for hm_img in hm_imgs:
            img_paths = []
            for attention_mode in attention_modes:
                img_paths.append(os.path.join(heatmap_path, attention_mode, cat, hm_img))
            for attention_mode in attention_modes:
                img_paths.append(os.path.join(ds_points_path, attention_mode, cat, hm_img.replace('.png', '_layer0.png')))
            
            fig, axes = plt.subplots(nrows=len(to_combine_paths), ncols=len(attention_modes), figsize=(len(attention_modes)*4, len(to_combine_paths)*4))
            save_path = os.path.join(save_cat_path, hm_img)
            for i_ax, ax in enumerate(axes.flat):
                image = plt.imread(img_paths[i_ax])
                ax.imshow(image)
                ax.axis('off')
                if i_ax < len(attention_modes):
                    ax.set_title(f'{attention_modes[i_ax]}')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            pbar.update(i_pbar)
            i_pbar += 1
    print(f'Done! All files are saved in {save_base_path}')
        
def visualize_modelnet_combine_prof(dir_name, asm="dot"):
    base_path = f'./artifacts/{dir_name}'
    heatmap_path = os.path.join(base_path, asm)
    # to_combine_paths = {
    #     'heatmap': heatmap_path,
    # }
    save_path = os.path.join(heatmap_path, f'vis_heatmap_compare_{asm}.png')
    
    idx_modes = os.listdir(heatmap_path)
    idx_modes.sort()
    selected_images = os.listdir(os.path.join(heatmap_path, idx_modes[0]))
    selected_images.sort()

    fig, axes = plt.subplots(nrows=len(selected_images), ncols=len(idx_modes), figsize=(len(idx_modes)*4, len(selected_images)*4))
    
    img_paths = []
    for selected_image in selected_images:
        for idx_mode in idx_modes:
            img_paths.append(os.path.join(heatmap_path, idx_mode, selected_image))
                    
    for i_ax, ax in enumerate(axes.flat):
        image = plt.imread(img_paths[i_ax])
        ax.imshow(image)
        ax.axis('off')
        if i_ax < len(idx_modes):
            ax.set_title(f'{idx_modes[i_ax]}')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
        
    print(f'Done!')   

def visualize_modelnet_downsampled_points_few_points(config, samples, index, cls_labels, mode, artifacts_path):
    M = config.test.few_points.num_points
    if M == 256:
        s_red = 2
    elif M == 128:
        s_red = 4
    elif M == 64:
        s_red = 6
    elif M == 32:
        s_red = 8
    elif M == 16:
        s_red = 12
    elif M == 8:
        s_red = 16
    elif M == 4:
        s_red = 24
    elif M == 2:
        s_red = 32
    else:
        s_red = 1
    base_path = f'{artifacts_path}/vis_ds_points/{mode}'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.downsample.M))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.downsample.M))]
    i_categories =[]
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    for i, (sample, category) in enumerate(zip(samples, categories)):
        
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category

        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        idx = index[0][i, 0]
        xyzRGB_tmp = deepcopy(xyzRGB)
        ds_tmp = []
        rst_tmp = []
        for each_idx in idx:
            xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
            ds_tmp.append(xyzRGB_tmp[each_idx])
        for each in xyzRGB_tmp:
            if each not in ds_tmp:
                rst_tmp.append(each)
        if config.test.visualize_downsampled_points.format == 'ply':
            saved_path = f'{cat_path}/{category}{i-i_saved}_layer0.ply'
            vertex = PlyElement.describe(np.array(xyzRGB_tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
            PlyData([vertex]).write(saved_path)
        elif config.test.visualize_downsampled_points.format == 'png':
            saved_path = f'{cat_path}/{category}{i-i_saved}_layer0.png'
            rst_vertex = np.array(rst_tmp)
            ds_vertex = np.array(ds_tmp)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-0.6, 0.6)
            ax.set_ylim3d(-0.6, 0.6)
            ax.set_zlim3d(-0.6, 0.6)
            ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o', s=1)
            ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:]/255, marker='o', s=s_red)
            plt.axis('off')
            plt.grid('off')
            plt.savefig(saved_path, bbox_inches='tight')
            plt.close(fig)
        else:
            raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')

def visualize_modelnet_downsampled_points_bin(config, samples, index, bin_prob, cls_labels, mode, artifacts_path):
    base_path = f'{artifacts_path}/vis_ds_points/{mode}'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
        bin_prob_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    i_categories =[]
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, (idx, bin_p) in enumerate(zip(index, bin_prob)):
            idx_tmp[layer].append(idx[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
            bin_prob_tmp[layer].append(bin_p[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    index = []
    bin_prob = []
    for (idx_each, bin_prob_each) in zip(idx_tmp, bin_prob_tmp):
        index.append(np.concatenate(idx_each))
        bin_prob.append(np.concatenate(bin_prob_each)) 
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    for i, (sample, category) in enumerate(zip(samples, categories)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category

        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)   
        txt_path = f'{cat_path}/{category}.txt'    
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            with open(txt_path, 'a') as f:
                row_str = f"{category}{i-i_saved}_layer{layer}:\t{', '.join([f'{num*100:.2f}%' for num in bin_prob[layer][i, 0]])}\n"
                f.write(row_str)
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer+1):
                    idx = index[layer-j][i, 0][idx]   # index mapping
            else:
                idx = index[layer][i, 0]
            
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i-i_saved}_layer{layer}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i-i_saved}_layer{layer}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-0.6, 0.6)
                ax.set_ylim3d(-0.6, 0.6)
                ax.set_zlim3d(-0.6, 0.6)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o', s=1)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:]/255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)             
    print(f'Done! All files are saved in {base_path}')
    
def visualize_shapenet_downsampled_points_bin(config, samples, index, bin_prob, cls_label, shape_ious, artifacts_path):
    base_path = f'{artifacts_path}/vis_ds_points'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    category_id_to_hash_code_mapping = {}
    for hash_code in list(config.datasets.mapping.keys()):
        category_id_to_hash_code_mapping[str(config.datasets.mapping[hash_code]['category_id'])] = hash_code
    categories = []
    for cat_id in cls_label:
        hash_code = category_id_to_hash_code_mapping[str(cat_id)]
        categories.append(config.datasets.mapping[hash_code]['category'])
    # select samples
    samples_tmp = []
    categories_tmp = []
    ious_tmp = []
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
        bin_p_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    i_categories =[]
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(np.asarray(categories)[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        ious_tmp.append(np.asarray(shape_ious)[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, (idx, bin_p) in enumerate(zip(index, bin_prob)):
            idx_tmp[layer].append(idx[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
            bin_p_tmp[layer].append(bin_p[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    shape_ious = np.concatenate(ious_tmp)
    index = []
    bin_prob = []
    for (idx_each, bin_p_each) in zip(idx_tmp, bin_p_tmp):
        index.append(np.concatenate(idx_each))
        bin_prob.append(np.concatenate(bin_p_each)) 
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    if config.datasets.dataset_name == "shapenet_AnTao350M":
        view_range = 0.6
    elif config.datasets.dataset_name == "shapenet_Yi650M" or config.datasets.dataset_name == "shapenet_Normal":
        view_range = 0.3
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    for i, (sample, category, iou) in enumerate(zip(samples, categories, shape_ious)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
            
        txt_path = f'{cat_path}/{category}.txt'    
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            with open(txt_path, 'a') as f:
                row_str = f"{category}{i-i_saved}_layer{layer}:\t{', '.join([f'{num*100:.2f}%' for num in bin_prob[layer][i, 0]])}\n"
                f.write(row_str)
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer+1):
                    idx = index[layer-j][i, 0][idx]   # index mapping
            else:
                idx = index[layer][i, 0]
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i-i_saved}_layer{layer}_{math.floor(iou * 1e5)}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i-i_saved}_layer{layer}_{math.floor(iou * 1e5)}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-view_range, view_range)
                ax.set_ylim3d(-view_range, view_range)
                ax.set_zlim3d(-view_range, view_range)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o', s=1)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')