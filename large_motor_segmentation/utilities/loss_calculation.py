import torch
from torch.nn import functional as F

from utilities.util import feature_transform_reguliarzer


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


def feature_transform_reguliarzer_pointnet(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
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

        if using_weight is False or using_weight == 0:
            loss = F.cross_entropy(pred, labels, reduction='mean')
        else:
            loss = F.cross_entropy(pred, labels, weight=weights, reduction='mean')

    return loss


def loss_calculation_pointnet(pred, target, trans_feat, weight=None, mat_diff_loss_scale=0.001):
    # loss = F.nll_loss(pred, target, weight=weight)
    target = target.type(torch.int64)
    loss = F.cross_entropy(pred, target, reduction='mean', weight=weight)
    mat_diff_loss = feature_transform_reguliarzer_pointnet(trans_feat)
    total_loss = loss + mat_diff_loss * mat_diff_loss_scale
    return total_loss


def loss_calculation_pointnet2(pred, target, weight=None):
    target = target.type(torch.int64)
    loss = F.cross_entropy(pred, target, reduction='mean', weight=weight)

    return loss
