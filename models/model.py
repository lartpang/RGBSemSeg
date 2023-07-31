import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .swin import model_cfgs, SwinTransformer


def rescale_2x(x: torch.Tensor, scale_factor=2, mode="bilinear", align_corners=False):
    return F.interpolate(
        x, scale_factor=scale_factor, mode=mode, align_corners=align_corners
    )


class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, mlp_ratio=1, bias=False, groups=1):
        super().__init__()
        self.groups = groups
        self.norm = nn.LayerNorm(in_dim)

        in_dim = groups * in_dim
        out_dim = groups * out_dim
        hid_dim = int(out_dim * mlp_ratio)
        self.fc1 = nn.Conv2d(in_dim, hid_dim, 1, bias=bias, groups=groups)
        self.dwconv = nn.Conv2d(
            hid_dim, hid_dim, kernel_size=3, padding=1, groups=hid_dim, bias=bias
        )
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_dim, out_dim, 1, bias=bias, groups=groups)

    def forward(self, x, h, w):
        x = self.norm(x)

        x = rearrange(x, "b (ng h w) c -> b (ng c) h w", ng=self.groups, h=h, w=w)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        x = rearrange(x, "b (ng c) h w -> b (ng h w) c", ng=self.groups)
        return x


class ConsistGuidedEnhancer(nn.Module):
    """Update information in x based on correlation between the part of x and the whole y"""

    def __init__(self, in_dim, hid_dim, num_heads, bias, k, mlp_ratio):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        # difference map extractor
        self.proj_2d_0 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hid_dim, bias)
        )
        self.proj_2d_1 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hid_dim, bias)
        )
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.merge = nn.Sequential(
            nn.LayerNorm(hid_dim * (num_heads + 1)),
            nn.Linear(hid_dim * (num_heads + 1), hid_dim, bias),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim, bias),
        )

        self.prototype_base = nn.Parameter(data=torch.ones(k, hid_dim))
        nn.init.trunc_normal_(self.prototype_base, std=0.02)

        self.q_proj0 = nn.Sequential(
            nn.LayerNorm(hid_dim), nn.Linear(hid_dim, hid_dim, bias=bias)
        )
        self.kv_proj0 = nn.Sequential(
            nn.LayerNorm(hid_dim), nn.Linear(hid_dim, hid_dim + hid_dim, bias=bias)
        )
        self.qkv_proj0 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.ffn0 = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, int(hid_dim * mlp_ratio), bias=bias),
            nn.GELU(),
            nn.Linear(int(hid_dim * mlp_ratio), hid_dim, bias=bias),
        )

        self.q_proj1 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hid_dim, bias=bias)
        )
        self.kv_proj1 = nn.Sequential(
            nn.LayerNorm(hid_dim), nn.Linear(hid_dim, hid_dim + hid_dim, bias=bias)
        )
        self.qkv_proj1 = nn.Linear(hid_dim, in_dim, bias=bias)
        self.ffn1 = FFN(in_dim, in_dim, mlp_ratio, bias=bias, groups=2)
        self.gamma = nn.Parameter(torch.zeros(size=(1, 1, in_dim)), requires_grad=True)

    def ops(self, x, y):
        return x * y

    def forward(self, x, y, H, W):
        """B,HW,C - B,HW,C - B,HW"""
        guide_feat = self.get_guide(x, y, H, W)  # B,HW,D

        # generated by prior
        kv = self.kv_proj0(guide_feat)
        k, v = kv.chunk(2, dim=-1)
        k = rearrange(k, "b hw (nh hd) -> b nh hw hd", nh=self.num_heads)
        v = rearrange(v, "b hw (nh hd) -> b nh hw hd", nh=self.num_heads)

        # initialize common prototype token based on x and y
        centers = self.q_proj0(self.prototype_base)  # K,C
        q = rearrange(centers, "k (nh hd) -> nh k hd", nh=self.num_heads)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        qk = torch.einsum("nkd, bnld -> bnkl", q, k)  # B,Nh,K,2HW

        qkv = qk.softmax(dim=-1) @ v  # B,Nh,K,D
        qkv = rearrange(qkv, "b nh k hd -> b k (nh hd)")

        # update centers themselves
        centers = self.qkv_proj0(qkv)  # B,K,C
        centers = self.ffn0(centers)  # B,K,C

        # update the cluster centers based on y
        xy = torch.cat([x, y], dim=1)  # B,2HW,C
        q = self.q_proj1(xy)  # B,2HW,C
        q = rearrange(q, "b hw (nh hd) -> b nh hw hd", nh=self.num_heads)
        kv = self.kv_proj1(centers)  # B,K,C
        kv = rearrange(kv, "b k (ng nh hd) -> ng b nh k hd", ng=2, nh=self.num_heads)
        k, v = kv[0], kv[1]

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        qk = q @ k.transpose(-1, -2)  # B,Nh,2HW,K

        qkv = qk.softmax(dim=-1) @ v  # B,Nh,2HW,D
        qkv = rearrange(qkv, "b nh hw hd -> b hw (nh hd)")  # B,2HW,C

        # remap and merge into x
        xy = xy + self.qkv_proj1(qkv) * self.gamma
        xy = xy + self.ffn1(xy, h=H, w=W)  # B,2HW,C

        x, y = torch.chunk(xy, 2, dim=1)
        guide_feat = self.minmax(guide_feat.mean(-1))
        return x, y, rearrange(guide_feat, "b (h w) -> b 1 h w", h=H, w=W)

    def get_guide(self, x, y, H, W):
        x_2d = self.proj_2d_0(x)
        y_2d = self.proj_2d_1(y)
        x_2d = rearrange(x_2d, "b (h w) c -> b c h w", h=H, w=W)
        y_2d = rearrange(y_2d, "b (h w) c -> b c h w", h=H, w=W)

        guide_feats = []
        for i in range(self.num_heads):
            guide_feats.append(self.ops(x_2d, y_2d))
            x_2d = self.pool(x_2d)
            y_2d = self.pool(y_2d)
        guide_feats.append(self.ops(x_2d, y_2d))
        guide_feat = torch.cat(guide_feats, dim=1)
        guide_feat = rearrange(guide_feat, "b c h w -> b (h w) c")

        guide_feat = self.merge(guide_feat)
        return guide_feat

    def minmax(self, guide_feat):
        guide_feat = guide_feat.detach()
        guide_feat = guide_feat - torch.min(guide_feat, dim=1, keepdim=True)[0]
        guide_feat = guide_feat / torch.max(guide_feat, dim=1, keepdim=True)[0]
        return guide_feat


class DiffGuidedEnhancer(nn.Module):
    """Update information in x based on correlation between the part of x and the whole y"""

    def __init__(self, in_dim, hid_dim, out_dim, num_heads, bias, k, mlp_ratio):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        # difference map extractor
        self.proj_2d_0 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hid_dim, bias)
        )
        self.proj_2d_1 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hid_dim, bias)
        )
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.merge = nn.Sequential(
            nn.LayerNorm(hid_dim * (num_heads + 1)),
            nn.Linear(hid_dim * (num_heads + 1), hid_dim, bias),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim, bias),
        )

        self.prototype_base = nn.Parameter(data=torch.ones(k, hid_dim))
        nn.init.trunc_normal_(self.prototype_base, std=0.02)

        self.q_proj0 = nn.Sequential(
            nn.LayerNorm(hid_dim), nn.Linear(hid_dim, hid_dim, bias=bias)
        )
        self.kv_proj0 = nn.Sequential(
            nn.LayerNorm(hid_dim), nn.Linear(hid_dim, hid_dim + hid_dim, bias=bias)
        )
        self.qkv_proj0 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.ffn0 = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, int(hid_dim * mlp_ratio), bias=bias),
            nn.GELU(),
            nn.Linear(int(hid_dim * mlp_ratio), hid_dim, bias=bias),
        )

        # self.channel_attn = DualChannel(in_dim=in_dim, out_dim=out_dim, num_heads=num_heads, bias=bias)
        self.merge_xy = FFN(2 * in_dim, out_dim=out_dim, bias=bias)
        self.merge_xyz = FFN(2 * out_dim, out_dim=out_dim, bias=bias)

        self.q_proj1 = nn.Sequential(
            nn.LayerNorm(out_dim), nn.Linear(out_dim, hid_dim, bias=bias)
        )
        self.kv_proj1 = nn.Sequential(
            nn.LayerNorm(hid_dim), nn.Linear(hid_dim, hid_dim + hid_dim, bias=bias)
        )
        self.qkv_proj1 = nn.Linear(hid_dim, out_dim, bias=bias)
        self.ffn1 = FFN(out_dim, out_dim=out_dim, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(size=(1, 1, out_dim)), requires_grad=True)

    def ops(self, x, y):
        return torch.abs(x - y)

    def forward(self, x, y, z, H, W):
        """B,HW,C"""
        guide_feat = self.get_guide(x, y, H, W)  # B,HW,1

        # generated by prior
        k, v = self.kv_proj0(guide_feat).chunk(2, dim=-1)
        k = rearrange(k, "b hw (nh hd) -> b nh hw hd", nh=self.num_heads)
        v = rearrange(v, "b hw (nh hd) -> b nh hw hd", nh=self.num_heads)

        # initialize common prototype token based on x and y
        centers = self.q_proj0(self.prototype_base)  # K,C
        q = rearrange(centers, "k (nh hd) -> nh k hd", nh=self.num_heads)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        qk = torch.einsum("nkd, bnld -> bnkl", q, k)  # B,Nh,K,HW

        qk = qk.softmax(dim=-1)
        qkv = torch.einsum("bnkl, bnld -> bnkd", qk, v)  # B,Nh,K,D
        qkv = rearrange(qkv, "b nh k hd -> b k (nh hd)")

        # update centers themselves
        centers = self.qkv_proj0(qkv)  # B,K,C
        centers = self.ffn0(centers)  # B,K,C

        # update the xy based on cluster centers
        xy = self.merge_xy(torch.cat([x, y], dim=-1), h=H, w=W)
        xyz = self.merge_xyz(torch.cat([xy, z], dim=-1), h=H, w=W)

        q = self.q_proj1(xyz)  # B,HW,C
        q = rearrange(q, "b hw (nh hd) -> b nh hw hd", nh=self.num_heads)
        kv = self.kv_proj1(centers)  # B,K,C
        kv = rearrange(kv, "b k (ng nh hd) -> ng b nh k hd", ng=2, nh=self.num_heads)
        k, v = kv[0], kv[1]

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        qk = q @ k.transpose(-1, -2)  # B,Nh,HW,K

        qkv = qk.softmax(dim=-1) @ v  # B,Nh,HW,D
        qkv = rearrange(qkv, "b nh hw hd -> b hw (nh hd)")  # B,HW,C

        # remap and merge into x
        xyz = xyz + self.qkv_proj1(qkv) * self.gamma
        xyz = xyz + self.ffn1(xyz, h=H, w=W)  # B,HW,C

        guide_feat = self.minmax(guide_feat.mean(dim=-1))
        return xyz, rearrange(guide_feat, "b (h w) -> b 1 h w", h=H, w=W)

    def get_guide(self, x, y, H, W):
        x_2d = self.proj_2d_0(x)
        y_2d = self.proj_2d_1(y)
        x_2d = rearrange(x_2d, "b (h w) c -> b c h w", h=H, w=W)
        y_2d = rearrange(y_2d, "b (h w) c -> b c h w", h=H, w=W)

        guide_feats = []
        for i in range(self.num_heads):
            guide_feats.append(self.ops(x_2d, y_2d))
            x_2d = self.pool(x_2d)
            y_2d = self.pool(y_2d)
        guide_feats.append(self.ops(x_2d, y_2d))
        guide_feat = torch.cat(guide_feats, dim=1)
        guide_feat = rearrange(guide_feat, "b c h w -> b (h w) c")

        guide_feat = self.merge(guide_feat)
        return guide_feat

    def minmax(self, guide_feat):
        guide_feat = guide_feat.detach()
        guide_feat = guide_feat - torch.min(guide_feat, dim=1, keepdim=True)[0]
        guide_feat = guide_feat / torch.max(guide_feat, dim=1, keepdim=True)[0]
        return guide_feat


class DualRGBModel(nn.Module):
    def __init__(
        self,
        k=4,
        mid_dim=64,
        num_classes=1,
        dec_num_heads=2,
        dec_mlp_ratio=1,
        use_checkpoint=False,
        pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth",
    ):
        super().__init__()
        # encoder
        model_cfg = model_cfgs["swin-b-384"]
        self.encoder = SwinTransformer(
            use_bchw=False,
            frozen_stages=-1,
            use_checkpoint=use_checkpoint,
            pretrained_url=pretrained if pretrained else "",
            **model_cfg,
        )

        # modality complementary interactive module
        self.comp_tr = nn.ModuleList()
        self.diff_tr = nn.ModuleList()
        for i_layer in range(self.encoder.num_layers):
            self.comp_tr.append(
                ConsistGuidedEnhancer(
                    in_dim=self.encoder.layers[i_layer].dim,
                    hid_dim=self.encoder.layers[i_layer].dim // 2,
                    k=k,
                    num_heads=dec_num_heads,
                    mlp_ratio=dec_mlp_ratio,
                    bias=False,
                )
            )
            self.diff_tr.append(
                DiffGuidedEnhancer(
                    in_dim=self.encoder.layers[i_layer].dim,
                    hid_dim=self.encoder.layers[i_layer].dim // 2,
                    out_dim=mid_dim,
                    k=k,
                    num_heads=dec_num_heads,
                    mlp_ratio=dec_mlp_ratio,
                    bias=False,
                )
            )

        self.deep = nn.Sequential(
            nn.LayerNorm(self.encoder.layers[3].dim * 2),
            nn.Linear(self.encoder.layers[3].dim * 2, mid_dim),
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, num_classes),
        )

    def body(self, image):
        H, W, img = self.encoder.stem(image)
        dep = img

        en_feats = []
        comp_maps = []
        for i in range(self.encoder.num_layers):
            encoder_layer = self.encoder.layers[i]
            img = encoder_layer(img, H, W)  # B,HW,C
            dep = encoder_layer(dep, H, W)  # B,HW,C

            img, dep, comp_map = self.comp_tr[i](img, dep, H, W)  # B,HW,C
            comp_maps.append(comp_map)

            en_feats.append((img, dep))  # B,HW,2C
            if encoder_layer.downsample is not None:
                img = encoder_layer.downsample(img, H, W)
                dep = encoder_layer.downsample(dep, H, W)
                H, W = (H + 1) // 2, (W + 1) // 2

        d4 = self.deep(torch.cat([img, dep], dim=-1))

        img, dep = en_feats[3]
        d3, diff_map = self.diff_tr[3](img, dep, d4, H, W)
        d3 = self.upsample(d3, h=H, w=W)
        H, W = H * 2, W * 2

        img, dep = en_feats[2]
        d2, diff_map = self.diff_tr[2](img, dep, d3, H, W)
        d2 = self.upsample(d2, h=H, w=W)
        H, W = H * 2, W * 2

        img, dep = en_feats[1]
        d1, diff_map = self.diff_tr[1](img, dep, d2, H, W)
        d1 = self.upsample(d1, h=H, w=W)
        H, W = H * 2, W * 2

        img, dep = en_feats[0]
        d0, diff_map = self.diff_tr[0](img, dep, d1, H, W)

        d0 = self.upsample(d0, h=H, w=W)
        H, W = H * 2, W * 2
        x = self.predictor(d0)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = rescale_2x(x, scale_factor=2)
        return {"logits": x, "comp_maps": comp_maps}

    def upsample(self, x, h, w, scale_factor=2):
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x = rescale_2x(x, scale_factor=scale_factor)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x

    def forward(self, image):
        output = self.body(image)
        return output["logits"]

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith("encoder.layers."):
                    param_groups["pretrained"].append(param)
                elif name.startswith("encoder"):
                    param.requires_grad = False
                    param_groups["fixed"].append(param)
                else:
                    param_groups["retrained"].append(param)
        print(
            f"Parameter Groups:{{\n"
            f"Pretrained: {len(param_groups['pretrained'])}\n"
            f"Fixed: {len(param_groups['fixed'])}\n"
            f"ReTrained: {len(param_groups['retrained'])}\n}}"
        )
        return param_groups

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, **kwargs):
        losses = []
        loss_str = []

        logits = all_preds["logits"]
        bce = F.binary_cross_entropy_with_logits(
            input=logits, target=gts, reduction="mean"
        )
        losses.append(bce)
        loss_str.append(f"bce: {bce.item():.5f}")
        return sum(losses), " ".join(loss_str)
