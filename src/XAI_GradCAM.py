import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import shap
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn



# ────────────────────────────────────────────────────────────────────────────────
# A) Grad‑CAM for Binary ViT Head
# ────────────────────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients   = None

        # 1) Forward hook to grab activations
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        target_layer.register_forward_hook(forward_hook)

        # 2) Backward hook to grab gradients of the layer's output
        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; grad_out[0] is the gradient w.r.t. the output
            self.gradients = grad_out[0].detach()
        target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, img_tensor):
        """
        img_tensor: [1,3,224,224] on device
        returns: heatmap [224,224,3] float in [0,1]
        """
        # Ensure gradients flow back to input
        img_tensor.requires_grad_(True)

        # 1) Forward
        logits = self.model(img_tensor)     # [1]

        # 2) Backward on the single logit
        self.model.zero_grad()
        logits[0].backward(retain_graph=True)

        # 3) Pool gradients: [C,H',W'] -> [C]
        grads = self.gradients[0]           # now populated
        weights = grads.mean(dim=(1,2))

        # 4) Weighted sum of activations -> CAM
        acts = self.activations[0]
        cam = F.relu((weights[:,None,None] * acts).sum(dim=0))

        # 5) Normalize to [0,1]
        cam -= cam.min()
        cam /= cam.max()

        # 6) Upsample & to 3‑channel
        cam_np = cam.cpu().numpy()
        cam_img = Image.fromarray((cam_np*255).astype(np.uint8)) \
                     .resize((224,224), Image.BILINEAR)
        #heatmap = np.stack([np.array(cam_img)/255]*3, axis=2)
        cam_resized = cv2.resize(cam_np, (224,224), interpolation=cv2.INTER_LINEAR)
        cmap = plt.get_cmap('jet')
        heatmap_rgba = cmap(cam_resized) 
        heatmap = heatmap_rgba[..., :3]


        return heatmap


