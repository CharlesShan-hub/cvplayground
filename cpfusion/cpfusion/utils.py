import torch 
import torch.nn.functional as F

def msd_align(images):
    def up_sample(img, target_shape):
        batch_size, channels, height, width = img.size()
        padded_img = torch.zeros(batch_size, channels, 2 * height, 2 * width, device=img.device)
        padded_img[:, :, ::2, ::2] = img
        padded_img[:, :, 1::2, ::2] = img
        padded_img[:, :, ::2, 1::2] = img
        padded_img[:, :, 1::2, 1::2] = img
        return padded_img[:,:,:target_shape[-2],:target_shape[-1]]
    
    if len(images) == 1:
        return images
    
    queue = [images[-1]]
    for i in range(len(images) - 1, 0, -1):
        for _ in range(len(images)-i):
            queue.insert(0,up_sample(queue.pop(), images[i-1].shape))
        queue.insert(0,images[i-1])
    return queue

def correlation_coefficient_weights(X, Y):
    # correlation
    cov_XY = torch.mean((X - X.mean(dim=(2, 3), keepdim=True)) * (Y - Y.mean(dim=(2, 3), keepdim=True)))

    # variation
    var_X = torch.var(X, dim=(2, 3), keepdim=True)
    var_Y = torch.var(Y, dim=(2, 3), keepdim=True)
    
    # correlation coefficient
    V_CC = cov_XY / torch.sqrt(var_X * var_Y)
    
    # weights
    W_CC = torch.where(V_CC > 0, 1 - torch.exp(-V_CC), torch.exp(V_CC) - 1)
    
    return W_CC

def base_layer_fuse(X, Y, wcc):
    weight = 1.5 ** ((wcc + 1) / 2)
    fused = torch.zeros_like(X)
    weighted_X = weight * X
    weighted_Y = (1 - weight) * Y
    fused[:, ::2, :, :] = torch.max(X, Y)[:, ::2, :, :]
    fused[:,1::2, :, :] = (weight * X + (1 - weight) * Y)[:,1::2, :, :]

    return fused.mean(dim=1,keepdim=True)

def calculate_SF(X):
    # Ensure the input is a 2D tensor
    assert X.dim() >= 2, "Input must be at least a 2D tensor."

    # Perform 2D FFT
    fftimage = torch.fft.fft2(X,dim=(-2,-1))

    # Compute the amplitude spectrum
    amplitudespectrum = torch.abs(fftimage)

    # Calculate the mean of the amplitude spectrum
    sf = torch.mean(amplitudespectrum,dim=(-2,-1))

    return sf

def split_img(X, n=3):
    def mirror_pad_if_odd_and_unfold(tensor):
        # padding
        pad_height = 0 if (tensor.shape[-2] % 2 == 0) else 1
        pad_width = 0 if (tensor.shape[-1] % 2 == 0) else 1
        tensor = F.pad(tensor, (0, pad_width, 0, pad_height),mode='constant')#, mode='reflect')
        
        # unfold
        H, W = tensor.shape[-2]//2, tensor.shape[-1]//2
        return torch.cat([
            tensor[..., :H, :W], # Top - Left
            tensor[..., :H,-W:], # Top - Right
            tensor[...,-H:, :W], # Bottom - Left
            tensor[...,-H:,-W:], # Bottom - Right
        ],dim=-3), (pad_height,pad_width)
    
    paddings = []
    for _ in range(n):
        X, pad = mirror_pad_if_odd_and_unfold(X)
        paddings.append(pad)
    return X,paddings

def merge_img(X, paddings, n=3):
    def fold(tensor,pad):
        (B,C,H,W) = tensor.shape
        C//=4
        padded = torch.zeros(size=(B,C,H*2,W*2))
        padded[:,:,:H,:W] = tensor[:,0*C:1*C,:,:]
        padded[:,:,:H,-W:] = tensor[:,1*C:2*C,:,:]
        padded[:,:,-H:,:W] = tensor[:,2*C:3*C,:,:]
        padded[:,:,-H:,-W:] = tensor[:,3*C:4*C,:,:]
        return padded[:,:,:H*2-pad[0],:W*2-pad[1]]

    for _ in range(n):
        X = fold(X, paddings.pop())
    return X

def detail_layer_fuse(X,Y):
    patchx, paddings = split_img(X)
    patchy, _ = split_img(Y)
    SF_X = calculate_SF(patchx)
    SF_Y = calculate_SF(patchy)
    alpha = (SF_X / (SF_X + SF_Y)).unsqueeze(-1).unsqueeze(-1)
    patchf = alpha * patchx + (1 - alpha) * patchy
    return merge_img(patchf, paddings).mean(dim=1,keepdim=True)

def test_split_and_merge():
    a1 = torch.rand(1, 1, 16, 16)
    a2 = torch.rand(2, 1, 16, 15)
    a3 = torch.rand(1, 3, 15, 14)
    a4 = torch.rand(8, 3, 16, 17)

    b1,p1 = split_img(a1)
    b2,p2 = split_img(a2)
    b3,p3 = split_img(a3)
    b4,p4 = split_img(a4)

    c1 = merge_img(b1,p1)
    c2 = merge_img(b2,p2)
    c3 = merge_img(b3,p3)
    c4 = merge_img(b4,p4)

    assert a1.equal(c1)
    assert a2.equal(c2)
    assert a3.equal(c3)
    assert a4.equal(c4)

if __name__ == '__main__':
    test_split_and_merge()