import click
from clib.utils import glance, rgb_to_gray
from clib.dataset.fusion import TNO, LLVIP
from clib.algorithms.msd import Laplacian
from clib.models import SimAMBlock
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from utils import *
from model import *

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
def main(**kwargs):
    dataset = LLVIP(root=kwargs['dataset_path'], transform=Compose([ToTensor()]), download=True, train=False)
    # dataset = TNO(root=kwargs['dataset_path'], transform=Compose([ToTensor()]), download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for ir,vis in dataloader:
        vis = rgb_to_gray(vis)
        # glance([ir,vis],title=['ir','vis'])

        # 多尺度分解
        ir_pyr = Laplacian(image = ir, layer = 4, gau_blur_way = 'Adaptive')
        vis_pyr = Laplacian(image = vis, layer = 4, gau_blur_way = 'Adaptive')

        # 基础层
        ir_base = ir_pyr.gaussian
        vis_base = vis_pyr.gaussian
        # glance([torch.abs(i) for i in ir_base])
        # glance([torch.abs(i) for i in vis_base])

        # 细节层
        ir_detail = ir_pyr.pyramid
        vis_detail = vis_pyr.pyramid
        # glance([torch.abs(i) for i in ir_detail])
        # glance([torch.abs(i) for i in vis_detail])

        # 上采样
        ir_detail = torch.cat(msd_align(ir_detail),dim=1)
        ir_base = torch.cat(msd_align(ir_base),dim=1)
        vis_detail = torch.cat(msd_align(vis_detail),dim=1)
        vis_base = torch.cat(msd_align(vis_base),dim=1)
        # glance(msd_align([torch.abs(i) for i in ir_pyr.pyramid]))
        # glance(msd_align([torch.abs(i) for i in ir_pyr.gaussian]))
        # glance(msd_align([torch.abs(i) for i in vis_pyr.pyramid]))
        # glance(msd_align([torch.abs(i) for i in vis_pyr.gaussian]))

        # 基础层融合
        # wcc = correlation_coefficient_weights(ir_base, vis_base)
        # fused_base = base_layer_fuse(ir_base, vis_base, wcc)
        fused_base = torch.max(torch.abs(ir_base),torch.abs(vis_base)).mean(dim=1,keepdim=True)
        glance([fused_base,torch.abs(ir_base).mean(dim=1,keepdim=True),torch.abs(vis_base).mean(dim=1,keepdim=True)])

        # 细节层增强
        # attension_block = SimAMBlock()
        attension_block = DSimAMBlock()
        # ir_detail_enhanced = ir_detail
        # vis_detail_enhanced = vis_detail
        ir_detail_enhanced = attension_block(ir_detail)
        vis_detail_enhanced = attension_block(vis_detail)

        # 细节层融合
        fuse_detail = detail_layer_fuse(ir_detail_enhanced, vis_detail_enhanced)

        # 细节层 + 基础层
        fused = fuse_detail + fused_base
        # glance([fused,fuse_detail,fused_base],auto_contrast=False)
        # glance([torch.abs(fuse_detail)+torch.abs(fused_base),fused,fuse_detail,fused_base],auto_contrast=False)
        glance([torch.abs(fuse_detail)+torch.abs(fused_base),torch.abs(fuse_detail),torch.abs(fused_base)])
        
        # breakpoint()

        # glance([ir, ir_pyr.recon] + [torch.abs(i) for i in ir_pyr.pyramid] + [ir_base] + [None, None, torch.abs(ir_detail_enhanced[:,:1,:,:]), torch.abs(ir_detail_enhanced[:,1:2,:,:]), torch.abs(ir_detail_enhanced[:,2:3,:,:]), torch.abs(ir_detail_enhanced[:,3:4,:,:])] + [None], title=['ir','recon','detail1','detail2','detail3','detail4','base layer',None,None,'detail1 PAM','detail2 PAM','detail3 PAM','detail4 PAM'],shape=(2,7))#,auto_contrast=False)

        # glance([ir,ir_detail_enhanced[:,:1,:,:],ir_detail[:,:1,:,:]],auto_contrast=False,title=['ir','Layer1 after PAM', 'Layer1 before PAM'])
        # glance([ir,ir_detail_enhanced[:,1:2,:,:],ir_detail[:,1:2,:,:]],auto_contrast=False,title=['ir','Layer2 after PAM', 'Layer2 before PAM'])
        # glance([ir,ir_detail_enhanced[:,2:3,:,:],ir_detail[:,2:3,:,:]],auto_contrast=False,title=['ir','Layer3 after PAM', 'Layer3 before PAM'])


if __name__ == '__main__':
    main()
