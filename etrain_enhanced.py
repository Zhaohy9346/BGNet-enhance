import torch
from torch.autograd import Variable
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
from net.bgnet_enhanced import Net_Enhanced
from enhancement.enhancement_net import EnhancementLoss
from utils.tdataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np

file = open("log/BGNet_Enhanced.txt", "a")
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch, enhancement_criterion):
    model.train()

    loss_record3 = AvgMeter()
    loss_record2 = AvgMeter()
    loss_record1 = AvgMeter()
    loss_recorde = AvgMeter()
    loss_record_enhance = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        # Data prepare
        images, gts, edges = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()

        # Forward
        lateral_map_3, lateral_map_2, lateral_map_1, edge_map, enhanced_img, filter_params = model(images)

        # COD losses
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss1 = structure_loss(lateral_map_1, gts)
        losse = dice_loss(edge_map, edges)

        # Enhancement loss (可选)
        if opt.use_enhancement and enhanced_img is not None:
            # Denormalize original image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            original_img = images * std + mean

            loss_enhance = enhancement_criterion(enhanced_img, original_img)
        else:
            loss_enhance = torch.tensor(0.0).cuda()

        # Total loss
        loss = loss3 + loss2 + loss1 + 3 * losse + loss_enhance

        # Backward
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        # Recording loss
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record1.update(loss1.data, opt.batchsize)
        loss_recorde.update(losse.data, opt.batchsize)
        loss_record_enhance.update(loss_enhance.data, opt.batchsize)

        # Train visualization
        if i % 60 == 0 or i == total_step:
            log_msg = ('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], '
                       '[edge: {:.4f}], [enhance: {:.4f}]'.format(
                datetime.now(), epoch, opt.epoch, i, total_step,
                loss_record3.avg, loss_record2.avg, loss_record1.avg,
                loss_recorde.avg, loss_record_enhance.avg))
            print(log_msg)
            file.write(log_msg + '\n')

    # Save checkpoint
    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 5 == 0 or (epoch + 1) == opt.epoch:
        torch.save(model.state_dict(), save_path + 'BGNet_Enhanced-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'BGNet_Enhanced-%d.pth' % epoch)
        file.write('[Saving Snapshot:]' + save_path + 'BGNet_Enhanced-%d.pth' % epoch + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=25, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=416, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str, default='BGNet_Enhanced')
    parser.add_argument('--use_enhancement', type=bool, default=True,
                        help='whether to use enhancement module')
    parser.add_argument('--enhancement_loss_weight', type=float, default=0.1,
                        help='weight for enhancement reconstruction loss')
    opt = parser.parse_args()

    # Build model
    model = Net_Enhanced(use_enhancement=opt.use_enhancement).cuda()

    # Enhancement loss
    enhancement_criterion = EnhancementLoss(weight=opt.enhancement_loss_weight)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root,
                              batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training with Enhancement Module")
    print(f"Use Enhancement: {opt.use_enhancement}")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch, enhancement_criterion)

    file.close()