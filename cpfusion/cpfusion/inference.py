import click
from clib.utils import *
from clib.algorithms.msd import Laplacian
from utils import *
from model import *
import copy

__all__ = [
    'fusion'
]

def c(image):
    B,_,H,W = image.shape
    res = torch.zeros(size=(B,3,H,W))
    res[:,0:1,:,:] = image
    res[:,1:3,:,:] = 128.0
    return ycbcr_to_rgb(res)

def fusion(ir, vis, layer, debug=False):
    # 得到亮度通道
    assert ir.shape[1] == 1 and ir.ndim == 4
    ir_y = rgb_to_ycbcr(gray_to_rgb(ir))[:,0:1,:,:]
    if vis.shape[1] == 1:
        vis = gray_to_rgb(vis)
    vis_ycbcr = rgb_to_ycbcr(vis)
    vis_y = vis_ycbcr[:,0:1,:,:]

    # 多尺度分解
    ir_pyr = Laplacian(image = ir_y, layer = layer, gau_blur_way = 'Adaptive')
    vis_pyr = Laplacian(image = vis_y, layer = layer, gau_blur_way = 'Adaptive')
    if debug:
        glance(
            [c(ir_pyr.recon), c(vis_pyr.recon), c(ir_y), c(vis_y)], 
            title=['IR rebuild', 'VIS rebuild', 'IR', 'VIS'],
            shape=(2,2), suptitle = 'Multi-Scale Decomposition (result)')

    # 基础层
    ir_base = ir_pyr.gaussian
    vis_base = vis_pyr.gaussian
    if debug:
        glance(
            [torch.abs(c(i)) for i in ir_base] + [torch.abs(c(i)) for i in vis_base],
            title=[f'Ir Base L{i+1}' for i in range(layer+1)] + [f'Vis Base L{i+1}' for i in range(layer+1)],
            shape=(2,layer+1),suptitle = 'Multi-Scale Decomposition (Base)'
        )

    # 细节层
    ir_detail = ir_pyr.pyramid
    vis_detail = vis_pyr.pyramid
    if debug:
        glance(
            [torch.abs(c(i)) for i in ir_detail] + [torch.abs(c(i)) for i in vis_detail],
            title=[f'Ir Detail L{i+1}' for i in range(layer)] + [f'Vis Detail L{i+1}' for i in range(layer)],
            shape=(2,layer),suptitle = 'Multi-Scale Decomposition (Detail)'
        )

    # 上采样 - 为了融合计算
    ir_detail = torch.cat(msd_align(ir_detail),dim=1)
    ir_base = torch.cat(msd_align(ir_base),dim=1)
    vis_detail = torch.cat(msd_align(vis_detail),dim=1)
    vis_base = torch.cat(msd_align(vis_base),dim=1)

    # 基础层融合 - wcc
    wcc = correlation_coefficient_weights(ir_base, vis_base)
    fused_base = base_layer_fuse(ir_base, vis_base, wcc)
    if debug:
        glance(
            [torch.abs(c(fused_base[:,i:i+1,:,:])) for i in range(layer)],
            title=[f'f_b{i+1} wcc={wcc[0,i:i+1,0,0].item()}' for i in range(layer)],
            suptitle = 'correlation coefficient weights'
        )
    
    # 细节层增强
    # attension_block = SimAMBlock()
    attension_block = DSimAMBlock()
    # ir_detail_enhanced = ir_detail
    # vis_detail_enhanced = vis_detail
    ir_detail_enhanced = attension_block(ir_detail)
    vis_detail_enhanced = attension_block(vis_detail)
    if debug:
        glance(
            [torch.abs(c(ir_detail[:,i:i+1,:,:])) for i in range(layer)]+\
            [torch.abs(c(ir_detail_enhanced[:,i:i+1,:,:])) for i in range(layer)]+\
            [torch.abs(c(vis_detail[:,i:i+1,:,:])) for i in range(layer)]+\
            [torch.abs(c(vis_detail_enhanced[:,i:i+1,:,:])) for i in range(layer)],
            title=[f'irl{i+1}' for i in range(layer)]+\
            [f'ir with Sim l{i+1}' for i in range(layer)]+\
            [f'vis l{i+1}' for i in range(layer)]+\
            [f'vis with Sim l{i+1}' for i in range(layer)],
            shape=(4,layer), suptitle = 'DSimAMBlock (Detail)', tight_layout=True
        )

    # 细节层融合
    fused_detail = detail_layer_fuse(ir_detail_enhanced, vis_detail_enhanced)

    # 下采样 - 重构图像
    fused_base = msd_resample(fused_base)
    fused_detail = msd_resample(fused_detail)
    fused_pyr = copy.deepcopy(ir_pyr)
    fused_pyr.gaussian = fused_base
    fused_pyr.pyramid = fused_detail
    fused_pyr.reconstruction()
    fused = copy.deepcopy(vis_ycbcr)
    fused[:,0:1,:,:] = fused_pyr.recon

    # 恢复成 RGB
    # glance(fused_pyr.recon,auto_contrast=True,clip=True)
    return ycbcr_to_rgb(fused).clip(max=1.0, min=0.0)
    
    # glance([fused,ir,vis],title=['fused','ir','vis'],auto_contrast=False,clip=True)

    # 细节层 + 基础层
    # fused = fused_detail + fused_base
    # glance([ir,vis,None,fused_detail+fused_base,fused_detail,fused_base],clip=True,auto_contrast=False, title=['ir','vis',None,'fused','fused detail', 'fused base'],shape=(2,3))

    # glance([ir, ir_pyr.recon] + [torch.abs(i) for i in ir_pyr.pyramid] + [ir_base] + [None, None, torch.abs(ir_detail_enhanced[:,:1,:,:]), torch.abs(ir_detail_enhanced[:,1:2,:,:]), torch.abs(ir_detail_enhanced[:,2:3,:,:]), torch.abs(ir_detail_enhanced[:,3:4,:,:])] + [None], title=['ir','recon','detail1','detail2','detail3','detail4','base layer',None,None,'detail1 PAM','detail2 PAM','detail3 PAM','detail4 PAM'],shape=(2,7))#,auto_contrast=False)

    # glance([ir,ir_detail_enhanced[:,:1,:,:],ir_detail[:,:1,:,:]],auto_contrast=False,title=['ir','Layer1 after PAM', 'Layer1 before PAM'])
    # glance([ir,ir_detail_enhanced[:,1:2,:,:],ir_detail[:,1:2,:,:]],auto_contrast=False,title=['ir','Layer2 after PAM', 'Layer2 before PAM'])
    # glance([ir,ir_detail_enhanced[:,2:3,:,:],ir_detail[:,2:3,:,:]],auto_contrast=False,title=['ir','Layer3 after PAM', 'Layer3 before PAM'])
    
    # save_array_to_img(fused=)
    return fused

@click.command()
@click.option("--layer", type=int, default=4)
def main(**kwargs):
    # ir = path_to_gray('/Users/kimshan/Public/project/paper/ir_250423.jpg')
    # vis = path_to_rgb('/Users/kimshan/Public/project/paper/vis_250423.jpg')
    # ir = path_to_gray('/Users/kimshan/Public/project/paper/ir_010379.jpg')
    # vis = path_to_rgb('/Users/kimshan/Public/project/paper/vis_010379.jpg')
    ir = path_to_gray('/Users/kimshan/Public/project/paper/ir_010001.jpg')
    vis = path_to_rgb('/Users/kimshan/Public/project/paper/vis_010001.jpg')
    
    ir = to_tensor(ir).unsqueeze(0)
    vis = to_tensor(vis).unsqueeze(0)

    glance([ir,vis,fusion(ir, vis, kwargs['layer'], debug=True)],title=['ir','vis','fused'],auto_contrast=False,clip=True)


if __name__ == '__main__':
    main()
