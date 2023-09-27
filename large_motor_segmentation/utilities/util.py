import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def cal_token_loss(bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals, ground_truth, args):
    # (B,2,T), (B,bolt_type,T), (B,3,T), (B,3,T), (B,2+bolt_type+3+3,T)

    loss = 0

    if not args.model.token.bolt_existing_loss == 0:
        logits = bolt_existing_label.permute(0, 2, 1)
        logits = logits.view(-1, args.num_segmentation_type).contiguous()
        # _                                                 (B,2,T) -> (B*T,2)
        target = ground_truth[:, 0:2].view(-1)
        target = target.type(torch.int64).contiguous()  # _     (B,T) -> (B*T)

        loss += args.model.token.bolt_existing_loss * F.cross_entropy(logits, target, reduction='mean')

    if not args.model.token.bolt_type_loss == 0:
        logits = bolt_type_pred.permute(0, 2, 1)
        logits = logits.view(-1, args.num_segmentation_type).contiguous()
        # _                                                 (B,bolt_type,T) -> (B*T,bolt_type)
        target = ground_truth[:, 2:2 + args.model.token.bolt_type].view(-1)
        target = target.type(torch.int64).contiguous()  # _     (B,T) -> (B*T)

        loss += args.model.token.bolt_type_loss * F.cross_entropy(logits, target, reduction='mean')
    if not args.model.token.bolt_centers_loss == 0:
        logits = bolt_centers
        target = ground_truth[:, 2 + args.model.token.bolt_type:2 + args.model.token.bolt_type + 3]

        pred = F.sigmoid(logits)
        loss += args.model.token.bolt_centers_loss * F.huber_loss(pred, target, delta=1)
    if not args.model.token.bolt_normals_loss == 0:
        logits = bolt_normals
        target = ground_truth[:, 2 + args.model.token.bolt_type + 3:2 + args.model.token.bolt_type + 3 + 3]

        pred = F.sigmoid(logits)
        loss += args.model.token.bolt_normals_loss * F.huber_loss(pred, target, delta=1)

    return loss


def cal_segment_loss(pred, target, weights, args):
    """
        Calculate cross entropy loss, apply label smoothing if needed.
        pred: (B, segment_type, N)
        target: (B, N)
    """

    pred = pred.permute(0, 2, 1)
    pred = pred.view(-1, args.num_segmentation_type).contiguous()
    # _                                                     (B,segment_type,N) -> (B*N,segment_type)
    target = target.view(-1)
    target = target.type(torch.int64).contiguous()  # _     (B,N) -> (B*N)

    if args.use_class_weight == 0:
        loss = F.cross_entropy(pred, target, reduction='mean')
    else:
        loss = F.cross_entropy(pred, target, weight=weights, reduction='mean')

    return loss


def cal_loss(point_segmentation_pred, target, weights, transform_matrix,
             bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals,
             args):
    segment_loss = cal_segment_loss(point_segmentation_pred, target, weights, args)
    token_loss = cal_token_loss(bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals, args)

    loss = segment_loss + token_loss

    if not args.stn_loss_weight == 0:
        loss += feature_transform_reguliarzer(transform_matrix) * args.stn_loss_weight

    return loss


def loss_calculation(pred, labels, weights=1, smoothing=False, using_weight=False):
    """
        Calculate cross entropy loss, apply label smoothing if needed.
        pred: (B*N, segment_type)
        target: (B*N)
    """
    #
    labels = labels.contiguous().view(-1)
    labels = labels.type(torch.int64)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        if using_weight:
            inter = -one_hot * log_prb

            loss = torch.matmul(inter, weights).sum(dim=1).mean()

        else:
            loss = -(one_hot * log_prb).sum(dim=1).mean()


    else:

        if using_weight:
            loss = F.cross_entropy(pred, labels, weight=weights, reduction='mean')
        else:
            loss = F.cross_entropy(pred, labels, reduction='mean')

    return loss


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    for b in range(B):
        pc = batch_data[b]
        centroid = torch.mean(pc, dim=0, keepdim=True)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1, keepdim=True)))
        pc = pc / m
        batch_data[b] = pc
    return batch_data


def rotate_per_batch(data, goals, angle_clip=np.pi * 1):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BXNx6 array, original batch of point clouds and point normals
        Return:
          BXNx3 array, rotated batch of point clouds
    """
    if goals != None:
        data = data.float()
        goals = goals.float()
        rotated_data = torch.zeros(data.shape, dtype=torch.float32)
        rotated_data = rotated_data.cuda()

        rotated_goals = torch.zeros(goals.shape, dtype=torch.float32).cuda()
        batch_size = data.shape[0]
        rotation_matrix = torch.zeros((batch_size, 3, 3), dtype=torch.float32).cuda()
        for k in range(data.shape[0]):
            angles = []
            for i in range(3):
                angles.append(random.uniform(-angle_clip, angle_clip))
            angles = np.array(angles)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(angles[0]), -np.sin(angles[0])],
                           [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                           [0, 1, 0],
                           [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                           [np.sin(angles[2]), np.cos(angles[2]), 0],
                           [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
            R = torch.from_numpy(R).float().cuda()
            rotated_data[k, :, :] = torch.matmul(data[k, :, :], R)
            rotated_goals[k, :, :] == torch.matmul(goals[k, :, :], R)
            rotation_matrix[k, :, :] = R
        return rotated_data, rotated_goals, rotation_matrix
    else:
        data = data.float()
        rotated_data = torch.zeros(data.shape, dtype=torch.float32)
        rotated_data = rotated_data.cuda()

        batch_size = data.shape[0]
        rotation_matrix = torch.zeros((batch_size, 3, 3), dtype=torch.float32).cuda()
        for k in range(data.shape[0]):
            angles = []
            for i in range(3):
                angles.append(random.uniform(-angle_clip, angle_clip))

            angles = np.array(angles)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(angles[0]), -np.sin(angles[0])],
                           [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                           [0, 1, 0],
                           [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                           [np.sin(angles[2]), np.cos(angles[2]), 0],
                           [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
            R = torch.from_numpy(R).float().cuda()

            rotated_data[k, :, :] = torch.matmul(data[k, :, :], R)

            rotation_matrix[k, :, :] = R
        return rotated_data, rotation_matrix


def feature_transform_reguliarzer(trans, GT=None):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    if GT == None:
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    else:
        loss = torch.mean(torch.norm(trans - GT, dim=(1, 2)))
    return loss


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def restructure_pretrained_checkpoint(checkpoint):
    model_state_dict = checkpoint['model_state_dict']
    for k, v in model_state_dict.items():
        print(k)
        if 'conv7.weight' in k:
            model_state_dict[k] = v[2:, :, :]
            print(model_state_dict[k].shape)
        if 'conv7.bias' in k:
            model_state_dict[k] = v[2:]
            print(model_state_dict[k].shape)
    checkpoint['model_state_dict'] = model_state_dict

    mIoU = checkpoint['mIoU']
    mIoU = mIoU / 6 * 8
    print(mIoU)
    checkpoint['mIoU'] = mIoU

    return checkpoint


def _pipeline_refactor_pretrained_checkpoint():
    import torch

    checkpoint = torch.load(r'D:\Jupyter\AgiProbot\large_motor_segmentation\best_m.pth')
    checkpoint = restructure_pretrained_checkpoint(checkpoint)

    torch.save(checkpoint, r'C:\Users\Lenovo\Desktop\best_m.pth')

    # checkpoint = torch.load(r'C:\Users\Lenovo\Desktop\best_m.pth')
    # refactor_pretrained_checkpoint(checkpoint)


def get_result_distribution_matrix(num_classes, predictions, groundtruth, xyticks=None, class_counts=None,
                                   show_plt=True,
                                   plt_save_dir=None):
    if class_counts is None:
        class_counts = torch.zeros(num_classes, num_classes)

        for i in range(num_classes):
            for j in range(num_classes):
                class_counts[i, j] = torch.sum((predictions == i) * (groundtruth == j)) / torch.sum(groundtruth == j)
    # class_counts = torch.log(class_counts)

    # Create a table
    fig, ax = plt.subplots()

    # Draw the table
    cax = ax.matshow(class_counts, cmap='viridis')

    # Set x and y axis labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # Set x and y axis tick labels
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    if xyticks is None:
        ax.set_xticklabels(np.arange(num_classes))
        ax.set_yticklabels(np.arange(num_classes))
    else:
        ax.set_xticklabels(xyticks, rotation=20, fontsize=7)
        ax.set_yticklabels(xyticks, fontsize=7)

    # Add a colorbar
    fig.colorbar(cax)

    # Show the table
    if show_plt:
        plt.show()

    if plt_save_dir is not None:
        plt.savefig(plt_save_dir)

        # Close the Matplotlib plot
    plt.close()


def save_tensorboard_log(IoUs, epoch, log_writer, mIoU, opt, train_loss, train_point_acc):
    log_writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch)
    log_writer.add_scalar('IoU_background/train_IoU_background', IoUs[0], epoch)
    log_writer.add_scalar('IoU_motor/train_IoU_motor', IoUs[1], epoch)
    log_writer.add_scalar('mIoU/train_mIoU', mIoU, epoch)
    log_writer.add_scalar('loss/train_loss', train_loss, epoch)
    log_writer.add_scalar('point_acc/train_point_acc', train_point_acc, epoch)
    print('Epoch %d, train loss: %.6f, train point acc: %.6f ' % (
        epoch, train_loss, train_point_acc))
    print('Train mean ioU %.6f' % mIoU)


if __name__ == "__main__":
    _pipeline_refactor_pretrained_checkpoint()
