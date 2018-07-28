import torch
from libs import res_unet_plus
model = res_unet_plus()
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
