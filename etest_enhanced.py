import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.bgnet_enhanced import Net_Enhanced
from utils.tdataloader import test_dataset
from utils.metrics import Fmeasure, MAE, Smeasure, Emeasure

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--pth_path', type=str,
                    default='/data1/zhy/BGNet-master/checkpoints/BGNet_Enhanced-v4/BGNet_Enhanced-24.pth')
parser.add_argument('--use_enhancement', type=bool, default=True,
                    help='whether to use enhancement module')
parser.add_argument('--save_results', type=bool, default=True,
                    help='whether to save prediction results')
parser.add_argument('--save_enhanced', type=bool, default=True,
                    help='whether to save enhanced images')
opt = parser.parse_args()

# 用于存储所有数据集的结果
all_results = {}

for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
    print(f"\n{'='*60}")
    print(f"Testing on {_data_name}...")
    print(f"{'='*60}")
    
    data_path = '/data1/zhy/lowlight-CODdatasets/TestDataset/{}'.format(_data_name)
    save_path = './results/BGNet_Enhanced-v4/{}/'.format(_data_name)

    # 加载模型
    model = Net_Enhanced(use_enhancement=opt.use_enhancement)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    # 创建保存目录
    if opt.save_results:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path + 'edge/', exist_ok=True)
        if opt.save_enhanced and opt.use_enhancement:
            os.makedirs(save_path + 'enhanced/', exist_ok=True)

    # 准备数据加载器
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    # 初始化评价指标
    FM = Fmeasure()
    SM = Smeasure()
    EM = Emeasure()
    MAE_metric = MAE()

    # 测试循环
    for i in tqdm(range(test_loader.size), desc=f"Processing {_data_name}"):
        image, gt, name = test_loader.load_data()
        
        # 处理ground truth
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        
        # 前向传播
        image = image.cuda()
        with torch.no_grad():
            _, _, res, e, enhanced_img, filter_params = model(image)

        # 上采样预测结果到原始尺寸
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # 保存COD预测结果（可选）
        if opt.save_results:
            imageio.imwrite(save_path + name, (res * 255).astype(np.uint8))

        # 保存增强图像（可选）
        if opt.save_enhanced and opt.use_enhancement and enhanced_img is not None:
            # enhanced_img 已经是 [0, 1] 范围的图像
            enhanced = enhanced_img.data.cpu().numpy().squeeze()  # (C, H, W)
            enhanced = np.transpose(enhanced, (1, 2, 0))  # (H, W, C) - CHW to HWC
            
            # 转换为 uint8
            enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
            
            # 调整到原始尺寸
            enhanced_pil = Image.fromarray(enhanced)
            enhanced_pil = enhanced_pil.resize(gt.shape[::-1], Image.BILINEAR)
            enhanced_pil.save(save_path + 'enhanced/' + name)

        # 转换为uint8格式用于评价指标计算
        pred_uint8 = (res * 255).astype(np.uint8)
        gt_uint8 = (gt * 255).astype(np.uint8)
        
        # 计算评价指标
        FM.step(pred=pred_uint8, gt=gt_uint8)
        SM.step(pred=res, gt=gt)  # Smeasure使用归一化的输入
        EM.step(pred=pred_uint8, gt=gt_uint8)
        MAE_metric.step(pred=res, gt=gt)  # MAE使用归一化的输入

    # 获取结果
    fm_results = FM.get_results()
    sm_results = SM.get_results()
    em_results = EM.get_results()
    mae_results = MAE_metric.get_results()

    # 整理结果
    results = {
        'Dataset': _data_name,
        'Smeasure': sm_results['sm'],
        'MAE': mae_results['mae'],
        'maxEm': em_results['em']['curve'].max(),
        'meanEm': em_results['em']['curve'].mean(),
        'adpEm': em_results['em']['adp'],
        'maxFm': fm_results['fm']['curve'].max(),
        'meanFm': fm_results['fm']['curve'].mean(),
        'adpFm': fm_results['fm']['adp'],
    }
    
    all_results[_data_name] = results

    # 打印当前数据集结果
    print(f"\n{_data_name} Results:")
    print(f"  Smeasure: {results['Smeasure']:.4f}")
    print(f"  MAE:      {results['MAE']:.4f}")
    print(f"  maxEm:    {results['maxEm']:.4f}")
    print(f"  meanEm:   {results['meanEm']:.4f}")
    print(f"  adpEm:    {results['adpEm']:.4f}")
    print(f"  maxFm:    {results['maxFm']:.4f}")
    print(f"  meanFm:   {results['meanFm']:.4f}")
    print(f"  adpFm:    {results['adpFm']:.4f}")

# 保存所有结果到文件
print(f"\n{'='*60}")
print("Saving results to file...")
print(f"{'='*60}")

results_file = open("test_results_BGNet_Enhanced-v4.txt", "w")
results_file.write("="*80 + "\n")
results_file.write(f"BGNet_Enhanced Test Results (use_enhancement={opt.use_enhancement})新损失函数\n")
results_file.write("="*80 + "\n\n")

for dataset_name, results in all_results.items():
    results_file.write(f"Dataset: {dataset_name}\n")
    results_file.write("-"*80 + "\n")
    results_file.write(f"  Smeasure:  {results['Smeasure']:.4f}\n")
    results_file.write(f"  MAE:       {results['MAE']:.4f}\n")
    results_file.write(f"  maxEm:     {results['maxEm']:.4f}\n")
    results_file.write(f"  meanEm:    {results['meanEm']:.4f}\n")
    results_file.write(f"  adpEm:     {results['adpEm']:.4f}\n")
    results_file.write(f"  maxFm:     {results['maxFm']:.4f}\n")
    results_file.write(f"  meanFm:    {results['meanFm']:.4f}\n")
    results_file.write(f"  adpFm:     {results['adpFm']:.4f}\n")
    results_file.write("\n")

results_file.write("="*80 + "\n")

# 计算平均结果
avg_results = {
    'Smeasure': np.mean([r['Smeasure'] for r in all_results.values()]),
    'MAE': np.mean([r['MAE'] for r in all_results.values()]),
    'maxEm': np.mean([r['maxEm'] for r in all_results.values()]),
    'meanEm': np.mean([r['meanEm'] for r in all_results.values()]),
    'adpEm': np.mean([r['adpEm'] for r in all_results.values()]),
    'maxFm': np.mean([r['maxFm'] for r in all_results.values()]),
    'meanFm': np.mean([r['meanFm'] for r in all_results.values()]),
    'adpFm': np.mean([r['adpFm'] for r in all_results.values()]),
}

results_file.write("Average Results Across All Datasets\n")
results_file.write("-"*80 + "\n")
results_file.write(f"  Smeasure:  {avg_results['Smeasure']:.4f}\n")
results_file.write(f"  MAE:       {avg_results['MAE']:.4f}\n")
results_file.write(f"  maxEm:     {avg_results['maxEm']:.4f}\n")
results_file.write(f"  meanEm:    {avg_results['meanEm']:.4f}\n")
results_file.write(f"  adpEm:     {avg_results['adpEm']:.4f}\n")
results_file.write(f"  maxFm:     {avg_results['maxFm']:.4f}\n")
results_file.write(f"  meanFm:    {avg_results['meanFm']:.4f}\n")
results_file.write(f"  adpFm:     {avg_results['adpFm']:.4f}\n")
results_file.write("="*80 + "\n")

results_file.close()

# 打印平均结果
print(f"\nAverage Results Across All Datasets:")
print(f"  Smeasure: {avg_results['Smeasure']:.4f}")
print(f"  MAE:      {avg_results['MAE']:.4f}")
print(f"  maxEm:    {avg_results['maxEm']:.4f}")
print(f"  meanEm:   {avg_results['meanEm']:.4f}")
print(f"  adpEm:    {avg_results['adpEm']:.4f}")
print(f"  maxFm:    {avg_results['maxFm']:.4f}")
print(f"  meanFm:   {avg_results['meanFm']:.4f}")
print(f"  adpFm:    {avg_results['adpFm']:.4f}")

print(f"\nResults saved to: test_results_BGNet_Enhanced.txt")
if opt.save_enhanced and opt.use_enhancement:
    print(f"Enhanced images saved to: ./results/BGNet_Enhanced/{{dataset}}/enhanced/")
print("Testing completed!")
