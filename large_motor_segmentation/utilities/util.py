import numpy as np
import torch
import torch.nn.functional as F


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


def get_result_distribution_matrix(num_classes, predictions, groundtruth,
                                   xyticks=None, class_counts=None,
                                   show_plt=True, plt_save_dir=None, sqrt_value=False):
    import matplotlib.pyplot as plt

    if class_counts is None:
        class_counts = torch.zeros(num_classes, num_classes)

        for i in range(num_classes):
            for j in range(num_classes):
                class_counts[i, j] = torch.sum((predictions == i) * (groundtruth == j)) / torch.sum(groundtruth == j)

    if sqrt_value:
        class_counts = torch.sqrt(class_counts)
        # class_counts = torch.log(class_counts)

    # Create a table
    fig, ax = plt.subplots()

    # Draw the table
    cax = ax.matshow(class_counts, cmap='viridis')

    # Set x and y axis labels
    if sqrt_value:
        ax.set_xlabel('Predicted(sqrt)')
    else:
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
        if sqrt_value:
            plt.savefig(plt_save_dir.split('.')[0] + '_sqrt.' + plt_save_dir.split('.')[1])
        else:
            plt.savefig(plt_save_dir)

        # Close the Matplotlib plot
    plt.close()


def save_tensorboard_log(IoUs, epoch, log_writer, mIoU, opt, loss, point_acc, class_acc, mode):
    if mode == 'train_binary':

        log_writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch)
        log_writer.add_scalar('IoU_background/train_IoU_background', IoUs[0], epoch)
        log_writer.add_scalar('IoU_motor/train_IoU_motor', IoUs[1], epoch)
        log_writer.add_scalar('mIoU/train_mIoU', mIoU, epoch)
        log_writer.add_scalar('loss/train_loss', loss, epoch)
        log_writer.add_scalar('point_acc/train_point_acc', point_acc, epoch)
        print('Epoch %d, train loss: %.6f, train point acc: %.6f ' % (
            epoch, loss, point_acc))
        print('Train mean ioU %.6f' % mIoU)
    elif mode == 'valid_binary':
        log_writer.add_scalar('loss/eval_loss', loss, epoch)
        log_writer.add_scalar('point_acc/eval_point_acc', point_acc, epoch)
        log_writer.add_scalar('point_acc/eval_class_acc', class_acc, epoch)
        log_writer.add_scalar('mIoU/eval_mIoU', mIoU, epoch)
        log_writer.add_scalar('IoU_background/eval_IoU_background', IoUs[0], epoch)
        log_writer.add_scalar('IoU_motor/eval_IoU_motor', IoUs[1], epoch)

        outstr = 'Epoch %d,  eval loss %.6f, eval point acc %.6f, eval avg class acc %.6f' % (epoch, loss,
                                                                                              point_acc,
                                                                                              class_acc)
        print(outstr)
        print('Valid mean ioU %.6f' % mIoU)
    elif mode == 'train_pipeline':

        log_writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch)

        log_writer.add_scalar('mIoU/train_mIoU', mIoU, epoch)
        log_writer.add_scalar('loss/train_loss', loss, epoch)
        log_writer.add_scalar('point_acc/train_point_acc', point_acc, epoch)
        log_writer.add_scalar('class_acc/train_class_acc', class_acc, epoch)

        log_writer.add_scalar('train_IoU/mIoU', mIoU, epoch)
        log_writer.add_scalar('train_IoU/Gear', IoUs[0], epoch)
        log_writer.add_scalar('train_IoU/Connector', IoUs[1], epoch)
        log_writer.add_scalar('train_IoU/Bolt', IoUs[2], epoch)
        log_writer.add_scalar('train_IoU/Solenoid', IoUs[3], epoch)
        log_writer.add_scalar('train_IoU/Electrical_Connector', IoUs[4], epoch)
        log_writer.add_scalar('train_IoU/Main_Housing', IoUs[5], epoch)

        print('Epoch %d, train loss: %.6f, train point acc: %.6f, eval avg class acc %.6f ' % (epoch,
                                                                                               loss,
                                                                                               point_acc,
                                                                                               class_acc))
        print('Train mean ioU %.6f' % mIoU)
    elif mode == 'valid_pipeline':
        log_writer.add_scalar('mIoU/eval_mIoU', mIoU, epoch)
        log_writer.add_scalar('loss/eval_loss', loss, epoch)
        log_writer.add_scalar('point_acc/eval_point_acc', point_acc, epoch)
        log_writer.add_scalar('point_acc/eval_class_acc', class_acc, epoch)

        log_writer.add_scalar('valid_IoU/mIoU', mIoU, epoch)
        log_writer.add_scalar('valid_IoU/Gear', IoUs[0], epoch)
        log_writer.add_scalar('valid_IoU/Connector', IoUs[1], epoch)
        log_writer.add_scalar('valid_IoU/Bolt', IoUs[2], epoch)
        log_writer.add_scalar('valid_IoU/Solenoid', IoUs[3], epoch)
        log_writer.add_scalar('valid_IoU/Electrical_Connector', IoUs[4], epoch)
        log_writer.add_scalar('valid_IoU/Main_Housing', IoUs[5], epoch)

        outstr = 'Epoch %d,  eval loss %.6f, eval point acc %.6f, eval avg class acc %.6f' % (epoch,
                                                                                              loss,
                                                                                              point_acc,
                                                                                              class_acc)
        print(outstr)
        print('Valid mean ioU %.6f' % mIoU)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    _pipeline_refactor_pretrained_checkpoint()
