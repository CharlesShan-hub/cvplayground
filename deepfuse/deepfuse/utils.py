from typing import Tuple
import torch

from clib.utils import *

def weightedFusion(
        cr1: torch.Tensor, 
        cr2: torch.Tensor, 
        cb1: torch.Tensor, 
        cb2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Perform the weighted fusing for Cb and Cr channel (paper equation 6)

        Arg:    cr1     (torch.Tensor)  - The Cr slice of 1st image
                cr2     (torch.Tensor)  - The Cr slice of 2nd image
                cb1     (torch.Tensor)  - The Cb slice of 1st image
                cb2     (torch.Tensor)  - The Cb slice of 2nd image
        Ret:    The fused Cr slice and Cb slice
    """
    # L1 norm
    L1_NORM = lambda b: torch.sum(torch.abs(b))

    # Fuse Cr channel
    cr_up = (cr1 * L1_NORM(cr1 - 127.5) + cr2 * L1_NORM(cr2 - 127.5))
    cr_down = L1_NORM(cr1 - 127.5) + L1_NORM(cr2 - 127.5)
    cr_fuse = cr_up / cr_down

    # Fuse Cb channel
    cb_up = (cb1 * L1_NORM(cb1 - 127.5) + cb2 * L1_NORM(cb2 - 127.5))
    cb_down = L1_NORM(cb1 - 127.5) + L1_NORM(cb2 - 127.5)
    cb_fuse = cb_up / cb_down

    return cr_fuse, cb_fuse
