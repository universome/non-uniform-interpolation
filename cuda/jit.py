from torch.utils.cpp_extension import load

gp_interp_cuda = load(
    'gp_interp_cuda', ['gp_interp_cuda.cpp', 'gp_interp_cuda_kernel.cu'], verbose=True)
# help(gp_interp_cuda)

import sys; sys.path.extend(['.'])

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TVF

from gp_interp import GPInterp

device = 'cuda'
img_pil = Image.open('/tmp/skoroki/datasets/ffhq/thumbnails128x128/00000.png')
img = TVF.to_tensor(img_pil).to(device)
gp = GPInterp(img.shape[1], img.shape[2], img.shape[0], 0.25, 1.0, 5)
gp.to(device)

print('Num coords:', len(gp.means))

print('Doing a forward pass...')
out = gp(img)

print('Doing a backward pass...')
loss = (out - img).abs().mean()
loss.backward()
print('Loss:', loss.item())
print('Success!')
