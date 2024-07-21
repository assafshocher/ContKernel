import torch
import matplotlib.pyplot as plt
import PIL
import torch.nn.functional as F
import imageio



def multires_rand_crop(lr, hr, crop_sz, sf_h=None, sf_w=None):
    # assumptions: 
    # lr and hr are 4D tensors with shape (N, C, H, W).
    # size of hr divides size of lr in both spatial dims.

    H_lr, W_lr = lr.shape[2], lr.shape[3]
    H_hr, W_hr = hr.shape[2], hr.shape[3]
    sf_h = H_hr // H_lr if sf_h is None else sf_h
    sf_w = W_hr // W_lr if sf_w is None else sf_w
    crop_sz_h, crop_sz_w = crop_sz if isinstance(crop_sz, tuple) else (crop_sz, crop_sz)

    # Randomly select the top left corner of the crop in the low-res image.
    # This is the center of the top-left pixel of the crop. and NOT the actual
    # top left corner of the crop.
    lr_crop_top_left_pixel_center_h = torch.randint(0, H_lr - crop_sz_h, (1,))
    lr_crop_top_left_pixel_center_w = torch.randint(0, W_lr - crop_sz_w, (1,))

    # Before subtracting 0.5 we have the center of the top-left pixel.
    # Then after subtracting 0.5 we have the top left corner of that pixel
    # which is the the true top left corner of the crop.
    # Now getting the crop center is easy. Sanity: for even sized crops the center is
    # on the boundary between pixels, so indexed as x.5. For odd sized crops the center
    # is a pixel center, so indexed as x.0.
    lr_crop_center_h = lr_crop_top_left_pixel_center_h - 0.5 + crop_sz_h / 2
    lr_crop_center_w = lr_crop_top_left_pixel_center_w - 0.5 + crop_sz_w / 2

    # Now we match the crop center to the high-res image using the equation from CC paper.
    # it is modified since we now the low-res is the source and the high-res is the target.
    hr_crop_center_h = lr_crop_center_h * sf_h + ((H_hr - 1) - (H_lr - 1) * sf_h) / 2
    hr_crop_center_w = lr_crop_center_w * sf_w + ((W_hr - 1) - (W_lr - 1) * sf_w) / 2

    # getting the center of top left corner pixel for the hr crop.
    hr_crop_top_left_pixel_center_h = hr_crop_center_h - crop_sz_h * sf_h / 2 + 0.5
    hr_crop_top_left_pixel_center_w = hr_crop_center_w - crop_sz_w * sf_w / 2 + 0.5

    # Now we can get the crops.
    lr_crop = lr[:, :, 
                int(lr_crop_top_left_pixel_center_h):
                int(lr_crop_top_left_pixel_center_h) + crop_sz_h, 
                int(lr_crop_top_left_pixel_center_w):
                int(lr_crop_top_left_pixel_center_w) + crop_sz_w]
    hr_crop = hr[:, :, 
                int(hr_crop_top_left_pixel_center_h):
                int(hr_crop_top_left_pixel_center_h) + crop_sz_h * sf_h, 
                int(hr_crop_top_left_pixel_center_w):
                int(hr_crop_top_left_pixel_center_w) + crop_sz_w * sf_w]
    
    return lr_crop, hr_crop


def imread(fname, bounds=(-1, 1),  **kwargs):
    image = PIL.Image.open(fname, **kwargs).convert(mode='RGB')
    tesnor = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())), dtype=torch.float32)
    tesnor = tesnor.view(image.size[1], image.size[0], len(image.getbands()))
    tesnor = tesnor.permute(2, 0, 1).unsqueeze(0) / 255.0
    vmin, vmax = bounds
    return torch.clamp((vmax - vmin) * tesnor + vmin, vmin, vmax)


def imshow(x, bounds=(-1, 1)):
    vmin, vmax = bounds
    x = x.detach().cpu()
    x = (x - vmin) / (vmax - vmin)
    if x.shape[0] == 1:
        x = x.squeeze(0)
    if x.shape[0] <= 3:
        x = torch.einsum('chw->hwc', x)
    if x.shape[-1] == 1:
        plt.imshow(x.squeeze(-1), cmap='gray')
    else:
        plt.imshow(x)
    plt.show()


def vis_kernel(k, k_shape, sf, sz=200, device=torch.device('cpu'), show=True):
    xy = torch.stack(torch.meshgrid([torch.linspace(-1, 1, sz, device=device) * k_shape[i] / 2
                                    for i in [0, 1]]))
    image = k(xy.view(2, -1, 1, 1), (1/sf[0], 1/sf[1])).view(sz, sz, 1)
    if show:
        imshow(image)
    return image


def vis_kernel_over_scales(k, k_shape, sfs, filename, sz=200, device=torch.device('cpu'), fps=10):
    frames = []
    for sf in sfs:
        frame = vis_kernel(k, k_shape, sf, sz=sz, device=device, show=False)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-12)
        frame = frame.repeat(1, 1, 3) * 255
        frame = F.interpolate(frame.transpose(-1,0).unsqueeze(0), size=(200, 200), mode='nearest').transpose(-1,1).squeeze(0)
        frames.append(frame.cpu().byte().numpy())
    imageio.mimsave(filename, frames, fps=fps)
    print(f'Saved {filename}!')