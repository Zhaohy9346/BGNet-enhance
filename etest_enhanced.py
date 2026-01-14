import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.bgnet_enhanced import Net_Enhanced
from utils.tdataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--pth_path', type=str,
                    default='./checkpoints/BGNet_Enhanced/BGNet_Enhanced-24.pth')
parser.add_argument('--use_enhancement', type=bool, default=True,
                    help='whether to use enhancement module')
parser.add_argument('--save_enhanced', type=bool, default=True,
                    help='whether to save enhanced images')

for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/BGNet_Enhanced/{}/'.format(_data_name)

    opt = parser.parse_args()
    model = Net_Enhanced(use_enhancement=opt.use_enhancement)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + 'edge/', exist_ok=True)
    if opt.save_enhanced:
        os.makedirs(save_path + 'enhanced/', exist_ok=True)

    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    print(f"Testing on {_data_name}...")
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        with torch.no_grad():
            _, _, res, e, enhanced_img, _ = model(image)

        # Save COD prediction
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path + name, (res * 255).astype(np.uint8))

        # Save enhanced image (optional)
        if opt.save_enhanced and enhanced_img is not None:
            enhanced = enhanced_img.data.cpu().numpy().squeeze()
            enhanced = np.transpose(enhanced, (1, 2, 0))  # CHW -> HWC
            enhanced = (enhanced * 255).astype(np.uint8)
            # Resize to original size
            from PIL import Image

            enhanced = Image.fromarray(enhanced)
            enhanced = enhanced.resize(gt.shape[::-1], Image.BILINEAR)
            enhanced.save(save_path + 'enhanced/' + name)

    print(f"Completed {_data_name}!")