import click
from typing import Union
import torch
from cslib.utils.image import to_tensor, rgb_to_ycbcr, gray_to_rgb, ycbcr_to_rgb, rgb_to_gray, save_array_to_img, path_to_gray, path_to_rgb
from cslib.utils import get_device, Options, glance
from cslib.algorithms.msd import Laplacian, Contrust
from cslib.models import SimAMBlock
from utils import *
import copy
from pathlib import Path
import torch.nn.functional as F

class ResSimAMBlock(torch.nn.Module):
    '''
    detail enhancement and feature highlighting module
    res + detail + SimAMBlock = ResSimAMBlock
    '''
    def __init__(self):
        self.res = 0.2
        super(ResSimAMBlock, self).__init__()
    
    def forward(self, X):
        # 计算图像的平均像素值
        mean_X = X.mean(dim=(2, 3), keepdim=True)
        
        # 计算每个像素与平均值的差的平方
        d = (X - mean_X) ** 2
        
        # 计算激活函数
        func = 1 / (1 + torch.exp(-d)) + 0.5
        
        # 应用激活函数到像素差异值，并保留30%原始信息
        X_enhance = X * func * (1-self.res) + X * self.res
        
        return X_enhance

from cslib.metrics.fusion import ag_metric, mi_metric, sf_metric, vif_metric, psnr_metric, ssim_metric, rmse_metric, q_abf_metric, vis

__all__ = [
    'fusion'
]

def _c(image: torch.Tensor) -> torch.Tensor:
    ''' 将灰度图转换为RGB图, 为了方便可视化
    '''
    B,_,H,W = image.shape
    res = torch.zeros(size=(B,3,H,W))
    res[:,0:1,:,:] = image
    res[:,1:3,:,:] = 128.0
    return to_tensor(ycbcr_to_rgb(res))

def image_init(ir: torch.Tensor, vis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ir.shape[1] == 1 and ir.ndim == 4 and vis.ndim == 4
    if vis.shape[1] == 1:
        vis = to_tensor(gray_to_rgb(vis))
    vis_ycbcr = to_tensor(rgb_to_ycbcr(vis))
    vis_y = vis_ycbcr[:, :1, :, :]
    ir_y = to_tensor(rgb_to_ycbcr(gray_to_rgb(ir)))[:, :1, :, :]
    return vis_ycbcr, vis_y, ir_y

def apple_msd(ir_y: torch.Tensor, vis_y: torch.Tensor, layer: int, msd_method: str, debug: bool) -> tuple[Union[Laplacian, Contrust], Union[Laplacian, Contrust]]:
    if msd_method == 'Laplacian':
        ir_pyr = Laplacian(image = ir_y, layer = layer, gau_blur_way = 'Adaptive', up_way = 'bilinear')
        vis_pyr = Laplacian(image = vis_y, layer = layer, gau_blur_way = 'Adaptive', up_way = 'bilinear')
    elif msd_method == 'Contrust':
        ir_pyr = Contrust(image = ir_y, layer = layer, gau_blur_way = 'Adaptive', up_way = 'bilinear')
        vis_pyr = Contrust(image = vis_y, layer = layer, gau_blur_way = 'Adaptive', up_way = 'bilinear')
    else:
        raise ValueError(f'Unknown msd method: {msd_method}')
    if debug:
        glance(
            [_c(ir_pyr.recon), _c(vis_pyr.recon), _c(ir_y), _c(vis_y)], 
            title=['IR rebuild', 'VIS rebuild', 'IR', 'VIS'],
            shape=(2,2), suptitle = 'Multi-Scale Decomposition (result)')
        # temp2_ir_pyr = copy.deepcopy(ir_pyr)
        # temp2_vis_pyr = copy.deepcopy(vis_pyr)
        # temp2_ir_pyr.up_way = 'bilinear'
        # temp2_vis_pyr.up_way = 'bilinear'
        # temp2_ir_pyr.reconstruction()
        # temp2_vis_pyr.reconstruction()
        # glance(
        #     [_c(temp2_ir_pyr.recon), _c(temp2_vis_pyr.recon) ,_c(ir_y), _c(vis_y)],
        # )
    return ir_pyr, vis_pyr

def get_base(ir_pyr: Union[Laplacian, Contrust], vis_pyr: Union[Laplacian, Contrust], layer: int, debug: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ir_base = ir_pyr.gaussian
    vis_base = vis_pyr.gaussian
    if debug:
        glance(
            [torch.abs(_c(i)) for i in ir_base] + [torch.abs(_c(i)) for i in vis_base],
            title=[f'Ir Base L{i+1}' for i in range(layer+1)] + [f'Vis Base L{i+1}' for i in range(layer+1)],
            shape=(2,layer+1),suptitle = 'Multi-Scale Decomposition (Base)'
        )
    ir_base = torch.cat(msd_align(ir_base),dim=1)
    vis_base = torch.cat(msd_align(vis_base),dim=1)
    return ir_base, vis_base

def get_detail(ir_pyr: Union[Laplacian, Contrust], vis_pyr: Union[Laplacian, Contrust], layer: int, debug: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ir_detail = ir_pyr.pyramid
    vis_detail = vis_pyr.pyramid
    if debug:
        glance(
            [torch.abs(_c(i)) for i in ir_detail] + [torch.abs(_c(i)) for i in vis_detail],
            title=[f'Ir Detail L{i+1}' for i in range(layer)] + [f'Vis Detail L{i+1}' for i in range(layer)],
            shape=(2,layer),suptitle = 'Multi-Scale Decomposition (Detail)'
        )
    
    ir_detail = torch.cat(msd_align(ir_detail),dim=1)
    vis_detail = torch.cat(msd_align(vis_detail),dim=1)
    return ir_detail, vis_detail

def base_layer_fuse(ir_base: torch.Tensor, vis_base: torch.Tensor, fusion_method: str, weight: torch.Tensor, layer: int, debug: bool) -> torch.Tensor:
    if fusion_method == 'SCENE':
        fused_base = ir_base * weight + vis_base * (1 - weight)
        if debug:
            glance(
                [torch.abs(_c(fused_base[:,i:i+1,:,:])) for i in range(layer)],
                title=[f'l{i+1}' for i in range(layer)],
                suptitle = 'Fused Base'
            )
    elif fusion_method == 'MAX':
        fused_base = torch.max(ir_base, vis_base)
        if debug:
            glance(
                [torch.abs(_c(fused_base[:,i:i+1,:,:])) for i in range(layer)],
                title=[f'f_b{i+1}' for i in range(layer)],
                suptitle = 'max'
            )
    else:
        raise ValueError(f'Unknown fusion method: {fusion_method}')
    return fused_base

def detail_layer_fuse(ir_detail: torch.Tensor, vis_detail: torch.Tensor, attension: str, layer: int, debug: bool) -> torch.Tensor:
    if attension != "None":
        if attension == 'SimAM':
            attension_block = SimAMBlock()  
        elif attension == 'ResSimAM':
            attension_block = ResSimAMBlock()
        else:
            raise ValueError(f'Unknown attension method: {attension}')
        ir_detail_enhanced = attension_block(ir_detail)
        vis_detail_enhanced = attension_block(vis_detail)
        if debug:
            glance(
                [torch.abs(_c(ir_detail[:,i:i+1,:,:])) for i in range(layer)]+\
                [torch.abs(_c(ir_detail_enhanced[:,i:i+1,:,:])) for i in range(layer)]+\
                [torch.abs(_c(vis_detail[:,i:i+1,:,:])) for i in range(layer)]+\
                [torch.abs(_c(vis_detail_enhanced[:,i:i+1,:,:])) for i in range(layer)],
                title=[f'irl{i+1}' for i in range(layer)]+\
                [f'ir with Sim l{i+1}' for i in range(layer)]+\
                [f'vis l{i+1}' for i in range(layer)]+\
                [f'vis with Sim l{i+1}' for i in range(layer)],
                shape=(4,layer), suptitle = 'Attension (Detail)', tight_layout=True
            )
    else:
        ir_detail_enhanced = ir_detail
        vis_detail_enhanced = vis_detail

    # 细节层融合
    # return ir_detail_enhanced * 0.5 + vis_detail_enhanced * 0.5
    return _detail_layer_fuse(ir_detail_enhanced, vis_detail_enhanced)

def reconstruction(fused_base: torch.Tensor, fused_detail: torch.Tensor, ir_pyr: Union[Laplacian, Contrust], vis_ycbcr: torch.Tensor) -> torch.Tensor:
    fused_base = msd_resample(fused_base)
    fused_detail = msd_resample(fused_detail)
    fused_pyr = copy.deepcopy(ir_pyr)
    fused_pyr.gaussian = fused_base
    fused_pyr.pyramid = fused_detail
    fused_pyr.reconstruction()
    fused = copy.deepcopy(vis_ycbcr)
    fused[:,0:1,:,:] = fused_pyr.recon
    return to_tensor(ycbcr_to_rgb(fused)).clip(max=1.0, min=0.0)

def saliency_map(img):
    img = rgb_to_gray(_c(img))[0,:,:,:]*255
    hist = torch.histc(img, bins=256, min=0, max=255)
    bins = torch.arange(256, device=img.device)
    sal_tab = torch.sum(hist * torch.abs(bins.unsqueeze(1) - bins), dim=1)
    saliency = sal_tab[img.long().flatten()].view_as(img)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())  # 归一化
    return saliency

def get_map(ir_pyr: Union[Laplacian, Contrust], vis_pyr: Union[Laplacian, Contrust], ir_y, vis_y, debug: bool) -> torch.Tensor:
    # 得到模糊图
    def get_blured(pyr):
        temp_pyr = copy.deepcopy(pyr)
        temp_pyr.pyramid = [torch.zeros_like(i) for i in temp_pyr.pyramid]
        temp_pyr.reconstruction()
        return temp_pyr.recon
    blured_ir = get_blured(ir_pyr)
    blured_vis = get_blured(vis_pyr)
    if debug:
        glance(
            [_c(blured_ir), _c(blured_vis), _c(ir_pyr.recon), _c(vis_pyr.recon), _c(ir_pyr.gaussian[-1]), _c(vis_pyr.gaussian[-1]),_c(ir_y), _c(vis_y)],
            title=['IR - base rebuild', 'VIS - base rebuild', 'IR - rebuild', 'VIS - rebuild', 'IR - base', 'VIS - base', 'IR', 'VIS'],
            shape=(2,4), suptitle = 'Multi-Scale Decomposition (result)'
        )

    # vis_saliency = saliency_map(vis_y).to(vis_y.device)
    # ir_saliency = saliency_map(ir_y).to(ir_y.device)
    vis_saliency = saliency_map(blured_vis).to(vis_y.device)
    ir_saliency = saliency_map(blured_ir).to(ir_y.device)
    if debug:
        glance(
            [vis_saliency, ir_saliency],
            title=['VIS - saliency map', 'IR - saliency map'],
            shape=(1,2), suptitle = 'Saliency Map'
        )
    
    # 得到权重图 ir明显，vis不明显的地方最重要
    weight = ir_saliency + 1 - vis_saliency
    weight = weight/weight.max()

    # 应用S曲线调整权重
    def contrast_curve(x, low=0.05, high=0.95, alpha=4.0):
        # 低于low的设为0，高于high的设为1
        mask_low = (x < low).float()
        mask_high = (x > high).float()
        mask_mid = 1 - mask_low - mask_high
        # 中间部分应用S曲线
        mid_part = 1 / (1 + torch.exp(-alpha * ((x - low)/(high-low) - 0.5)))
        
        return mask_high + mask_mid * mid_part
    enhanced_weight = contrast_curve(weight)

    if debug:
        glance(
            [weight, enhanced_weight],
            title=['Weight', 'Enhanced Weight'],
            shape=(1,2), suptitle = 'Weight Map'
        )

    return enhanced_weight.squeeze(0)

def fusion(
        ir: torch.Tensor, 
        vis: torch.Tensor, 
        layer: int = 4, 
        debug: bool = False,
        msd_method: str = ['Laplacian','Contrust'][0],
    ) -> torch.Tensor:
    # 得到Y通道
    vis_ycbcr, vis_y, ir_y = image_init(ir, vis)

    # 多尺度分解
    ir_pyr, vis_pyr = apple_msd(ir_y, vis_y, layer, msd_method, debug)

    # 基础层+对齐
    ir_base, vis_base = get_base(ir_pyr, vis_pyr, layer, debug)

    # 细节层+对齐
    ir_detail, vis_detail = get_detail(ir_pyr, vis_pyr, layer, debug)

    # 细节层融合
    # fused_detail = (ir_detail + vis_detail)/2
    # fused_detail = torch.max(ir_detail, vis_detail)
    # fused_detail = detail_layer_fuse(ir_detail, vis_detail, 'SimAM', layer, debug)
    fused_detail = detail_layer_fuse(ir_detail, vis_detail, 'ResSimAM', layer, debug)

    # 基础层融合
    # fused_base = (ir_base + vis_base)/2
    weight = get_map(ir_pyr, vis_pyr, ir_y, vis_y, debug)
    fused_base = base_layer_fuse(ir_base, vis_base, 'SCENE', weight, layer, debug)

    # 重构图像 - 下采样 + 恢复成 RGB
    fused = reconstruction(fused_base, fused_detail, ir_pyr, vis_ycbcr)
    
    return fused

@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][1])
@click.option("--device", type=str, default='auto')
def main(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('SceneFuse', kwargs)
    opts.present()
    
    # 黑天 - 保留红外细节
    image_index = ['00274','00388'][1] 
    # image_index = ['00031','01227'][0] # 白天 - 保留红外细节, 可见光部分，光污染
    # image_index = ['00385','00712','00716'][0] # 白天 - 保留红外细节, 可见光部分雾气污染

    ir = path_to_gray(f'{opts.p}/m3fd/fusion/ir/{image_index}.png')
    vis = path_to_rgb(f'{opts.p}/m3fd/fusion/vis/{image_index}.png')
    
    ir = to_tensor(ir).unsqueeze(0).to(opts.device)
    vis = to_tensor(vis).unsqueeze(0).to(opts.device)
    fused = fusion(ir, vis, kwargs['layer'], debug=True)

    print(f"AG: {ag_metric(ir, vis, fused)}")
    print(f"MI: {mi_metric(ir, vis, fused)}")
    print(f"SF: {sf_metric(ir, vis, fused)}")
    print(f"VIF: {vif_metric(ir, vis, fused)}")
    print(f"PSNR: {psnr_metric(ir, vis, fused)}")
    print(f"SSIM: {ssim_metric(ir, vis, fused)}")
    print(f"RMSE: {rmse_metric(ir, vis, fused)}")
    print(f"Q_ABF: {q_abf_metric(ir, vis, fused)}")

    glance([ir,vis,fused],title=['ir','vis','fused'],auto_contrast=False,clip=True)
    # save_array_to_img(fusion(ir, vis, kwargs['layer'], debug=False), filename=f'/Volumes/Charles/data/vision/torchvision/tno/tno/fused/cpfusion/{image_index}.png')


@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/tno/tno")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_m3fd(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion TNO', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'ir').glob("*.png"):
        ir = to_tensor(path_to_gray(i)).to(opts.device)
        vis = path_to_rgb(Path(opts.p) / 'vis' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False)
        name = Path(opts.p) / 'fused' / 'cpfusion' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)


@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/tno/tno")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_tno(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion TNO', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'ir').glob("*.png"):
        ir = to_tensor(path_to_gray(i)).to(opts.device)
        vis = path_to_rgb(Path(opts.p) / 'vis' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False)
        name = Path(opts.p) / 'fused' / 'cpfusion' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)


@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/llvip")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_llvip(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion LLVIP', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'infrared' / 'test').glob("*.jpg"):
        ir = path_to_gray(i)
        vis = path_to_rgb(Path(opts.p) / 'visible' / 'test' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False)
        name = Path(opts.p) / 'fused' / 'cpfusion' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)

def test_m3df():
    dataset = [
        '00025.png','00353.png','00443.png','00606.png','00826.png','00922.png','01115.png',
        '00177.png','00370.png','00449.png','00633.png','00829.png','00926.png','01122.png',
        '00196.png','00385.png','00453.png','00712.png','00834.png','00950.png','01136.png',
        '00202.png','00386.png','00461.png','00716.png','00857.png','00958.png','01156.png',
        '00232.png','00388.png','00479.png','00762.png','00867.png','00965.png','01165.png',
        '00284.png','00389.png','00489.png','00787.png','00871.png','00976.png','01186.png',
        '00325.png','00409.png','00497.png','00801.png','00878.png','00994.png','01204.png',
        '00334.png','00421.png','00512.png','00805.png','00896.png','01017.png','01212.png',
        '00339.png','00434.png','00525.png','00818.png','00910.png','01034.png',
        '00349.png','00441.png','00527.png','00825.png','00916.png','01043.png',
    ]
    dataset_path = '/Users/kimshan/Public/data/vision/torchvision'
    save_path = '/Users/kimshan/Public/data/vision/torchvision/m3fd/lunwen/SceneFuse'
    for i in range(len(dataset)):
    
        ir_path = Path(dataset_path) / 'm3fd/fusion/ir' / dataset[i]
        vis_path = Path(dataset_path) / 'm3fd/fusion/vis' / dataset[i]
        fused = Path(save_path) / dataset[i]

        ir = path_to_gray(ir_path)
        vis = path_to_rgb(vis_path)
        ir = to_tensor(ir).unsqueeze(0).to('mps')
        vis = to_tensor(vis).unsqueeze(0).to('mps')
        fused = fusion(ir, vis, 4, debug=False)
        name = Path(save_path) / dataset[i]
        save_array_to_img(fused, name, True)


if __name__ == '__main__':
    # main()
    # test_tno()
    # test_llvip()
    test_m3df()
