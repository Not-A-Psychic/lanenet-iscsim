import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import logging
from datetime import datetime
from tqdm import tqdm

from torch.nn.parallel.scatter_gather import gather
from dataloader import get_data_loader
from models.model import LaneNet
from models.loss import DiscriminativeLoss
from utils.utils import AverageMeter, adjust_learning_rate
from utils.metrics import batch_pix_accuracy, batch_intersection_union
from utils.parallel import DataParallelModel
from models.hnet import compute_hnet_loss

logger = logging.getLogger(__name__)


def train(opt, model, criterion_disc, criterion_ce, optimizer, loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    pbar = tqdm(loader)
    for data in pbar:
        data_time.update(time.time() - end)

        images, bin_labels, ins_labels, n_lanes = data

        images = Variable(images)
        bin_labels = Variable(bin_labels)
        ins_labels = Variable(ins_labels)

        if torch.cuda.is_available():
            images = images.cuda()
            bin_labels = bin_labels.cuda()
            ins_labels = ins_labels.cuda()
            n_lanes = n_lanes.cuda()

        if torch.cuda.device_count() <= 1:
            bin_preds, ins_preds = model(images)
        else:
            bin_preds, ins_preds = gather(model(images), 0, dim=0)

        _, bin_labels_ce = bin_labels.max(1)
        ce_loss = criterion_ce(
            bin_preds.permute(0, 2, 3, 1).contiguous().view(-1, 2),
            bin_labels_ce.view(-1))

        if ins_labels.unique().numel() <= 1:
            print("All instance labels are zero, skipping discriminative loss")
            disc_loss = torch.tensor(0.0, requires_grad=True, device=ins_preds.device)
        else:
            disc_loss = criterion_disc(ins_preds, ins_labels, n_lanes)

        if torch.isnan(disc_loss):
            print("disc_loss was NaN, replacing with 0")
            disc_loss = torch.tensor(0.0, requires_grad=True, device=ins_preds.device)

        loss = ce_loss + disc_loss

        if torch.isnan(loss):
            print("NaN detected in loss")
            print("CE loss:", ce_loss.item())
            print("Disc loss:", disc_loss.item())
            exit()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)

        pbar.set_description(
            '>>> Training loss={:.6f}, i/o time={data_time.avg:.3f}s, gpu time={batch_time.avg:.3f}s'.format(
                loss.item(),
                data_time=data_time,
                batch_time=batch_time))
        end = time.time()


def test(opt, model, criterion_disc, criterion_ce, loader):
    val_loss = AverageMeter()
    model.eval()

    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    pbar = tqdm(loader)

    with torch.no_grad():
        for data in pbar:
            images, bin_labels, ins_labels, n_lanes = data

            images = Variable(images)
            bin_labels = Variable(bin_labels)
            ins_labels = Variable(ins_labels)

            if torch.cuda.is_available():
                images = images.cuda()
                bin_labels = bin_labels.cuda()
                ins_labels = ins_labels.cuda()
                n_lanes = n_lanes.cuda()

            if torch.cuda.device_count() <= 1:
                bin_preds, ins_preds = model(images)
            else:
                bin_preds, ins_preds = gather(model(images), 0, dim=0)

            _, bin_labels_ce = bin_labels.max(1)
            ce_loss = criterion_ce(
                bin_preds.permute(0, 2, 3, 1).contiguous().view(-1, 2),
                bin_labels_ce.view(-1))

            disc_loss = criterion_disc(ins_preds, ins_labels, n_lanes)
            loss = ce_loss + disc_loss

            val_loss.update(loss.item())

            correct, labeled = batch_pix_accuracy(bin_preds.data, bin_labels[:,1,:,:])
            inter, union = batch_intersection_union(bin_preds.data, bin_labels[:,1,:,:], 2)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            pbar.set_description(
                '>>> Validating loss={:.6f}, pixAcc={:.6f}, mIoU={:.6f}'.format(
                    loss.item(), pixAcc, mIoU))

    return val_loss


def main(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    else:
        torch.manual_seed(opt.seed)

    train_loader = get_data_loader(opt,
                                   split='train',
                                   return_org_image=False)

    val_loader = get_data_loader(opt,
                                 split='val',
                                 return_org_image=False)

    output_dir = os.path.dirname(opt.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info('Building model...')

    model = LaneNet(cnn_type=opt.cnn_type, embed_dim=opt.embed_dim)
    model = DataParallelModel(model)

    criterion_disc = DiscriminativeLoss(delta_var=0.5,
                                        delta_dist=1.5,
                                        norm=2,
                                        usegpu=True)

    criterion_ce = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    if opt.start_from:
        logger.info('Restart training from %s', opt.start_from)
        checkpoint = torch.load(opt.start_from)
        model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        criterion_disc.cuda()
        criterion_ce.cuda()
        model = model.cuda()

    logger.info("Start training...")
    best_loss = sys.maxsize
    best_epoch = 0

    for epoch in tqdm(range(opt.num_epochs), desc='Epoch: '):
        learning_rate = adjust_learning_rate(opt, optimizer, epoch)
        logger.info('===> Learning rate: %f: ', learning_rate)

        train(opt, model, criterion_disc, criterion_ce, optimizer, train_loader)

        if epoch % opt.val_step == 0:
            val_loss = test(opt, model, criterion_disc, criterion_ce, val_loader)
            logger.info('Val loss: %s\n', val_loss)

            loss = val_loss.avg
            if loss < best_loss:
                logger.info(
                    'Found new best loss: %.7f, previous loss: %.7f',
                    loss,
                    best_loss)
                best_loss = loss
                best_epoch = epoch

                logger.info('Saving new checkpoint to: %s', opt.output_file)
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'opt': opt
                }, opt.output_file)

            else:
                logger.info(
                    'Current loss: %.7f, best loss is %.7f @ epoch %d',
                    loss,
                    best_loss,
                    best_epoch)

        if epoch - best_epoch > opt.max_patience:
            logger.info('Terminated by early stopping!')
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('meta_file', type=str, help='path to the metadata file containing train/val/test splits and image locations')
    parser.add_argument('output_file', type=str, help='output model file (*.pth)')
    parser.add_argument('--start_from', type=str, help='model to start from file (*.pth)')
    parser.add_argument('--dataset', default='tusimple', choices=['tusimple', 'culane', 'bdd', 'iscwebots'], help='Name of dataset')
    parser.add_argument('--image_dir', type=str, help='path to image dir')
    parser.add_argument('--cnn_type', default='unet', choices=['unet', 'unetscnn', 'deeplab'], help='The CNN used for image encoder')
    parser.add_argument('--embed_dim', type=int, default=4, help='Size of the lane embeddings')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--width', type=int, default=512, help='image width to the network')
    parser.add_argument('--height', type=int, default=256, help='image height to the network')
    parser.add_argument('--loader_type', type=str, choices=['dataset', 'dirloader', 'tusimpletest'], default='dataset', help='data loader type, dir: from a directory; meta: from a metadata file')
    parser.add_argument('--thickness', type=int, default=5, help='thickness of the polylines')
    parser.add_argument('--max_lanes', type=int, default=5, help='max number of lanes')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='max number of epochs to run the training')
    parser.add_argument('--lr_update', default=50, type=int, help='Number of epochs to update the learning rate.')
    parser.add_argument('--max_patience', type=int, default=5, help='max number of epoch to run since the minima is detected -- early stopping')
    parser.add_argument('--val_step', type=int, default=1, help='how often do we check the model (in terms of epoch)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers (each worker use a process to load a batch of data)')
    parser.add_argument('--log_step', type=int, default=20, help='How often to print training info (loss, system/data time, etc)')
    parser.add_argument('--use_hnet', action='store_true', help='Option to apply H-Net')
    parser.add_argument('--seed', type=int, default=123, help='random number generator seed to use')

    opt = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info('Input arguments: %s', json.dumps(vars(opt), sort_keys=True, indent=4))

    start = datetime.now()
    main(opt)
    logger.info('Time: %s', datetime.now() - start)

