import argparse
import os
import torch
import yaml
import numpy as np
import torch.nn.functional as F
import cv2


import config_folder as cf
from data_loaders.Chairs import Chairs
from data_loaders.kitti import KITTI
from data_loaders.sintel import Sintel
from data_loaders.KLens import KLens
from frame_utils import writeFlow
import flow_viz
from model import MaskFlownet, MaskFlownet_S, Upsample, EpeLossWithMask


def disparity_writeout(disp, path_ref, path_meas, mask):
    # Warping
    root_path = "./out_images/"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    writeFlow(
        os.path.join(
            root_path,
            os.path.basename(
                os.path.splitext(path_ref)[0]
            ) +
            "_" +
            os.path.basename(
                os.path.splitext(path_meas)[0]
            ) +
            ".flo"),
        disp
    )
    cv2.imwrite(
        os.path.join(
            root_path,
            os.path.basename(
                os.path.splitext(path_ref)[0]
            ) +
            "_" +
            os.path.basename(
                os.path.splitext(path_meas)[0]
            ) +
            "_flow.jpg"
        ),
        flow_viz.flow_to_image(disp)[:, :, [2, 1, 0]]
    )
    cv2.imwrite(
        os.path.join(
            root_path,
            os.path.basename(
                os.path.splitext(path_ref)[0]
            ) +
            "_" +
            os.path.basename(
                os.path.splitext(path_meas)[0]
            ) +
            "_mask.png"
        ),
        mask*255
    )


def centralize(img1, img2):
    rgb_mean = torch.cat((img1, img2), 2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, -1).mean(2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, 1, 1)
    return img1 - rgb_mean, img2-rgb_mean, rgb_mean


parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, nargs='?', default=None)
parser.add_argument('--dataset_cfg', type=str, default='chairs.yaml')
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='model checkpoint to load')
parser.add_argument('-b', '--batch', type=int, default=1,
                    help='Batch Size')
parser.add_argument('-f', '--root_folder', type=str, default=None,
                    help='Root folder of KITTI')
parser.add_argument('--resize', type=str, default='')
args = parser.parse_args()
resize = (int(args.resize.split(',')[0]), int(
    args.resize.split(',')[1])) if args.resize else None
num_workers = 2

print(os.path.join('config_folder', args.dataset_cfg))

with open(os.path.join('config_folder', args.dataset_cfg)) as f:
    config = cf.Reader(yaml.load(f))
with open(os.path.join('config_folder', args.config)) as f:
    config_model = cf.Reader(yaml.load(f))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = eval(config_model.value['network']['class'])(config)
checkpoint = torch.load(os.path.join('weights', args.checkpoint))

net.load_state_dict(checkpoint)
net = net.to(device)

if config.value['dataset'] == 'kitti':
    dataset = KITTI(args.root_folder, split='train',
                    editions='mixed', resize=resize, parts='valid')
elif config.value['dataset'] == 'chairs':
    dataset = Chairs(args.root_folder, split='valid')
elif config.value['dataset'] == 'sintel':
    dataset = Sintel(args.root_folder, split='valid', subset='final')
elif config.value['dataset'] == 'klens':
    dataset = KLens()
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          shuffle=False,
                                          batch_size=args.batch,
                                          num_workers=num_workers,
                                          drop_last=False,
                                          pin_memory=True)

epe = []
for idx, sample in enumerate(data_loader):
    with torch.no_grad():
        im0, im1, label, mask, path,raftflow = sample
        if config.value['dataset'] == 'klens':
            im0_path = path[0]
            im1_path = path[1]

        im0 = im0.permute(0, 3, 1, 2)
        im1 = im1.permute(0, 3, 1, 2)
        im0, im1, _ = centralize(im0, im1)

        shape = im0.shape
        pad_h = (64 - shape[2] % 64) % 64
        pad_w = (64 - shape[3] % 64) % 64
        if pad_h != 0 or pad_w != 0:
            im0 = F.interpolate(
                im0, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
            im1 = F.interpolate(
                im1, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')

        im0 = im0.to(device)
        im1 = im1.to(device)
        print(im0_path)
        print(im1_path)

        pred, flows, warpeds = net(im0, im1, raftflow, im0_path[0], im1_path[0])

        up_flow = Upsample(pred[-1], 4)
        up_occ_mask = Upsample(flows[0], 4)

    if pad_h != 0 or pad_w != 0:
        up_flow = F.interpolate(up_flow, size=[shape[2], shape[3]], mode='bilinear') * \
            torch.tensor([shape[d] / up_flow.shape[d]
                          for d in (2, 3)], device=device).view(1, 2, 1, 1)
        up_occ_mask = F.interpolate(
            up_occ_mask, size=[shape[2], shape[3]], mode='bilinear')

    print("left : ",im0_path[0], "right : ",im1_path[0])
    if config.value['dataset'] == 'klens':
        for i in range(up_flow.shape[0]):
            disparity_writeout(
                up_flow[i].permute(1, 2, 0).cpu().numpy(),
                im0_path[i],
                im1_path[i],
                up_occ_mask[i].permute(1, 2, 0).cpu().numpy(),
            )

    #epe.append(EpeLossWithMask()(up_flow, label, mask).detach())

    # Flip the flow to get the final prediction
    #final_flow = up_flow.flip(1)


print("\n\nCheck out_images for output!\n\n")
