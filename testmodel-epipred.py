from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset2
from pointnet.model import PointNetDenseCls4, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score,recall_score,roc_curve,auc


parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--batchSize', type=int, default=16, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--npoints', type=int, default=3000, help='subsample points')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
# parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--class_choice', type=str, default='protein', help="class_choice")
parser.add_argument('--r', type=str, default='recept', help="recept_choice")
parser.add_argument('--l', type=str, default='ligand', help="ligand_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

test_dataset_r = ShapeNetDataset2(
    root=opt.dataset,
    npoints=3000,
    classification=False,
    class_choice=[opt.r],
    split='test',
    data_augmentation=False)
testdataloader_r = torch.utils.data.DataLoader(
    test_dataset_r,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

test_dataset_l = ShapeNetDataset2(
    root=opt.dataset,
    npoints=3000,
    classification=False,
    class_choice=[opt.l],
    split='test',
    data_augmentation=False)
testdataloader_l = torch.utils.data.DataLoader(
    test_dataset_l,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

num_classes = test_dataset_l.num_seg_classes
classifier = PointNetDenseCls4(k=num_classes, feature_transform=opt.feature_transform)
classifier.cuda()

# PATH='/dartfs-hpc/rc/home/w/f00355w/Bdai/pointprotein/masif54-bce-nosigmoid/seg_model_protein_68.pth'
# PATH='/dartfs-hpc/rc/home/w/f00355w/Bdai/pointprotein/dbdb80/seg_model_protein_106.pth'
# PATH='/dartfs-hpc/rc/home/w/f00355w/Bdai/pointprotein/fine_tune80/seg_model_protein_24.pth'
PATH='/dartfs-hpc/rc/home/w/f00355w/Bdai/pointprotein/fine_tune801/seg_model_protein_76.pth'
# PATH='/dartfs-hpc/rc/home/w/f00355w/Bdai/pointprotein/dbd86/seg_model_protein_157.pth'
# PATH='/dartfs-hpc/rc/home/w/f00355w/Bdai/pointprotein/fine_tune802/seg_model_protein_73.pth'
#142  129 122
#39 45 52 57-59 67 73 76

classifier.load_state_dict(torch.load(PATH))
classifier.eval()
# print(len(testdataloader_r))
# j, (datar,datal) = next(enumerate(zip(testdataloader_r,testdataloader_l), 0))
all=[]
allp=[]
allr=[]
allauc=[]
epip=[]
epir=[]
epiauc=[]
for j, (datar,datal) in enumerate(zip(testdataloader_r,testdataloader_l), 0):
    pointsr, targetr = datar
    pointsl, targetl = datal
    # print(pointsr.size())
    # memlim = 110000
    memlim=90000
    if pointsl.size()[1] + pointsr.size()[1] > memlim:
        print(pointsl.size()[1] + pointsr.size()[1])
        lr = pointsl.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
        rr = pointsr.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
        ls = np.random.choice(pointsl.size()[1], lr, replace=False)
        rs = np.random.choice(pointsr.size()[1], rr, replace=False)

        # while targetr[:, rs].sum()==0 or targetl[:, ls].sum()==0:
        #     ls = np.random.choice(pointsl.size()[1], lr, replace=False)
        #     rs = np.random.choice(pointsr.size()[1], rr, replace=False)

        pointsr = pointsr[:, rs, :]
        targetr = targetr[:, rs]
        pointsl = pointsl[:, ls, :]
        targetl = targetl[:, ls]
    pointsr = pointsr.transpose(2, 1)
    pointsl = pointsl.transpose(2, 1)
    pointsr, targetr = pointsr.cuda(), targetr.cuda()
    pointsl, targetl = pointsl.cuda(), targetl.cuda()
    classifier = classifier.eval()
    try:
        pred, _, _ = classifier(pointsr,pointsl)
    except:
        continue
    pred = pred.view(-1, 1)
    target=torch.cat((targetr,targetl),1)
    target = target.view(-1, 1) - 1
    loss = F.binary_cross_entropy_with_logits(pred, target.float(), pos_weight=torch.FloatTensor([(target.size()[0]-float(target.cpu().sum()))*1.0/float(target.cpu().sum())]).cuda())
    # loss = F.binary_cross_entropy_with_logits(pred, target.float(), pos_weight=torch.FloatTensor([(target.size()[0]-float(target.cpu().sum()))*1.0/float(target.cpu().sum())]).cuda())

    pred_choice = torch.gt(torch.sigmoid(pred.data), 0.5).long()

    correct0 = pred_choice.eq(target.data).cpu().sum()
    correct1 = (pred_choice.eq(target.data).long() * target.data).cpu().sum()
    epoch=0
    num_batch=0
    i=0
    blue = lambda x: '\033[94m' + x + '\033[0m'
    # print(targetr.size())
    # print(target.size())
    # print(pointsr.size())
    # print(pred_choice.size())
    # print(target[1:targetr.size()[1],:])
    # if j==0:
    #     np.savetxt('/dartfs-hpc/rc/lab/C/CBKlab/Bdai/pointprotein/l.pts',pointsl.view(3,-1).cpu())
    #     np.savetxt('/dartfs-hpc/rc/lab/C/CBKlab/Bdai/pointprotein/r.pts', pointsr.view(3, -1).cpu())
    #     np.savetxt('/dartfs-hpc/rc/lab/C/CBKlab/Bdai/pointprotein/tar.seg', target.view(1, -1).cpu())
    #     np.savetxt('/dartfs-hpc/rc/lab/C/CBKlab/Bdai/pointprotein/pred.seg', pred_choice.view(1, -1).cpu())
    print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, j, num_batch, blue('test'), loss.item(), correct0.item()/float(opt.batchSize * target.size()[0])))
    all.append(correct0.item()/float(opt.batchSize * target.size()[0]))
    print('[%d: %d/%d] %s loss: %f class 1 accuracy: %f' % (epoch, j, num_batch, blue('test'), loss.item(), correct1.item() / float(target.data.sum().item())))

    print('[%d: %d/%d] %s loss: %f presicion: %f' % (epoch, j, num_batch, blue('test'), loss.item(), precision_score(target.data.cpu(),pred_choice.cpu())))
    allp.append(precision_score(target.data.cpu(),pred_choice.cpu()))
    print('[%d: %d/%d] %s loss: %f recall: %f' % (epoch, j, num_batch, blue('test'), loss.item(), recall_score(target.data.cpu(),pred_choice.cpu())))
    allr.append(recall_score(target.data.cpu(),pred_choice.cpu()))
    fpr, tpr, thresholds = roc_curve(target.data.cpu(),
                                     torch.sigmoid(pred.data).cpu(), pos_label=1)
    print('[%d: %d/%d] %s loss: %f auc: %f' % (epoch, j, num_batch, blue('test'), loss.item(), auc(fpr, tpr)))
    allauc.append(auc(fpr, tpr))
    print(float(target.data.sum().item())/float(opt.batchSize * target.size()[0]))

    print('[%d: %d/%d] %s loss: %f presicion: %f' % (epoch, j, num_batch, blue('test'), loss.item(), precision_score(target[1:targetr.size()[1],:].data.cpu(),pred_choice[1:targetr.size()[1],:].cpu())))
    epip.append(precision_score(target[1:targetr.size()[1],:].data.cpu(),pred_choice[1:targetr.size()[1],:].cpu()))
    print('[%d: %d/%d] %s loss: %f recall: %f' % (epoch, j, num_batch, blue('test'), loss.item(), recall_score(target[1:targetr.size()[1],:].data.cpu(),pred_choice[1:targetr.size()[1],:].cpu())))
    epir.append(recall_score(target[1:targetr.size()[1],:].data.cpu(),pred_choice[1:targetr.size()[1],:].cpu()))
    # print(target[1:targetr.size()[1],:].data.cpu())
    # print(torch.sigmoid(pred.data)[1:targetr.size()[1],:].cpu())
    fpr, tpr, thresholds = roc_curve(target[1:targetr.size()[1],:].data.cpu(), torch.sigmoid(pred.data)[1:targetr.size()[1],:].cpu(), pos_label=1)
    print('[%d: %d/%d] %s loss: %f auc: %f' % (epoch, j, num_batch, blue('test'), loss.item(), auc(fpr, tpr)))
    epiauc.append(auc(fpr, tpr))

print(len(all))
print(sum(all)*1.0/len(all))
print(sum(allp)*1.0/len(all))
print(sum(allr)*1.0/len(all))
print(sum(allauc)*1.0/len(all))
print('epi\n')
print(sum(epip)*1.0/len(all))
print(sum(epir)*1.0/len(all))
print(sum(epiauc)*1.0/len(all))