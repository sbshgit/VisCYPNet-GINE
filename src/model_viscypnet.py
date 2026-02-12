


import torch
from torch import nn
import timm

# ────────────────────────────────────────────────────────────────────────────────
# 1) Task‑Specific Head: CYPHead
# ────────────────────────────────────────────────────────────────────────────────
class CYPHead(nn.Module):
    
    def __init__(self, in_dim, hidden_dims=[256, 64], dropout=0.2):

        """
        A small MLP that sits on top of the frozen ViT backbone.

        in_dim       : dimensionality of the CLS embedding from the backbone
        hidden_dims  : list of hidden layer sizes, e.g. [512, 128]
        dropout      : dropout probability after each hidden layer
        """
        super().__init__()
        layers = []
        # Normalize the input features (stabilizes training)
        layers.append(nn.LayerNorm(in_dim))

        prev_dim = in_dim
        # Build each hidden block: Linear → Activation → Dropout
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),     
                nn.ReLU(inplace=True),      
                nn.Dropout(dropout)        
            ]
            prev_dim = h

        # Final binary output layer 
        layers.append(nn.Linear(prev_dim, 1))

        # Combine into a single nn.Sequential
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [batch_size, in_dim]
        returns: [batch_size] raw logits (no sigmoid applied here)
        """
        return self.mlp(x).squeeze(1)  # remove trailing dim


# ────────────────────────────────────────────────────────────────────────────────
# 2) Full Model: VisCYPNet
# ────────────────────────────────────────────────────────────────────────────────
class VisCYPNet(nn.Module):
    def __init__(self,
                 backbone=None,
                 backbone_name="vit_base_patch16_224",
                 pretrained=True,
                 head_hidden_dims=[256,64],
                 head_dropout=0.2,
                 device="cuda"):
        """
        Combines a frozen ViT backbone with a trainable CYPHead.

        backbone        : if provided, use this nn.Module as the feature extractor
        backbone_name   : otherwise load this model from timm (ViT by name)
        pretrained      : whether to fetch pretrained weights from timm
        head_hidden_dims: hidden layer sizes for CYPHead
        head_dropout    : dropout rate in CYPHead
        device          : torch device (for dummy forwarding if needed)
        """
        super().__init__()

        # ────────────────────────────────────────────────────────────────────────
        # Backbone loading or injection
        # ────────────────────────────────────────────────────────────────────────
        if backbone is not None:
            # Use  already‑wrapped CLIP visual module
            self.backbone = backbone
        else:
            # Otherwise, load a standard ViT (e.g. ViT-B/16) with no classification head
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0   # get raw CLS embedding output
            )

        # Freeze all backbone parameters (we only train the head)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ────────────────────────────────────────────────────────────────────────
        # Infer feature dimension (feat_dim) dynamically
        # ────────────────────────────────────────────────────────────────────────
        # timm ViTs have .num_features; CLIP models may use .output_dim;
        # otherwise we do a dummy forward to discover the output size.
        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        elif hasattr(self.backbone, "output_dim"):
            feat_dim = self.backbone.output_dim
        else:
            # If using a custom CLIP wrapper, check for .visual.output_dim
            vis = getattr(self.backbone, "visual", self.backbone)
            if hasattr(vis, "output_dim"):
                feat_dim = vis.output_dim
            else:
                # Fallback: run a dummy input through to measure dimension
                self.backbone.to(device)
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 224, 224, device=device)
                    out = self.backbone(dummy)
                feat_dim = out.shape[-1]

        # ────────────────────────────────────────────────────────────────────────
        # Attach the trainable head
        # ────────────────────────────────────────────────────────────────────────
        self.head = CYPHead(
            in_dim=feat_dim,
            hidden_dims=head_hidden_dims,
            dropout=head_dropout
        )

    def forward(self, x):
        """
        x: [batch_size, 3, 224, 224] tensor of images
        returns: [batch_size] logits from the MLP head
        """
        feats  = self.backbone(x)      # extract CLS features
        logits = self.head(feats)      # predict inhibition logit
        return logits




































































