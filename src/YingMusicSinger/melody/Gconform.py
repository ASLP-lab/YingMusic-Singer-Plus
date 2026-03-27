import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)

        return out * gate.sigmoid()


class conform_conv(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 31, DropoutL=0.1, bias: bool = True
    ):
        super().__init__()
        self.act2 = nn.SiLU()
        self.act1 = GLU(1)

        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0:
        #    it's a causal convolution, the input will be padded with
        #    `self.lorder` frames on the left in forward (causal conv impl).
        # else: it's a symmetrical convolution

        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        self.norm = nn.BatchNorm1d(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.drop = nn.Dropout(DropoutL) if DropoutL > 0.0 else nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.act1(self.pointwise_conv1(x))
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.act2(x)
        x = self.pointwise_conv2(x)
        return self.drop(x).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, conditiondim=None):
        super().__init__()
        if conditiondim is None:
            conditiondim = dim

        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(conditiondim, hidden_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(
                hidden_dim,
                dim,
            ),
        )

    def forward(self, q, kv=None, mask=None):
        # b, c, h, w = x.shape
        if kv is None:
            kv = q
        # q, kv = map(
        #     lambda t: rearrange(t, "b c t -> b t c", ), (q, kv)
        # )

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=2)

        q, k, v = map(
            lambda t: rearrange(t, "b t (h c) -> b h t c", h=self.heads), (q, k, v)
        )

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = rearrange(
            out,
            "b h t c -> b t (h c) ",
            h=self.heads,
        )
        return self.to_out(out)


class conform_ffn(nn.Module):
    def __init__(self, dim, DropoutL1: float = 0.1, DropoutL2: float = 0.1):
        super().__init__()
        self.ln1 = nn.Linear(dim, dim * 4)
        self.ln2 = nn.Linear(dim * 4, dim)
        self.drop1 = nn.Dropout(DropoutL1) if DropoutL1 > 0.0 else nn.Identity()
        self.drop2 = nn.Dropout(DropoutL2) if DropoutL2 > 0.0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.ln2(x)
        return self.drop2(x)


class conform_blocke(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,
        conv_drop: float = 0.1,
        ffn_latent_drop: float = 0.1,
        ffn_out_drop: float = 0.1,
        attention_drop: float = 0.1,
        attention_heads: int = 4,
        attention_heads_dim: int = 64,
    ):
        super().__init__()
        self.ffn1 = conform_ffn(dim, ffn_latent_drop, ffn_out_drop)
        self.ffn2 = conform_ffn(dim, ffn_latent_drop, ffn_out_drop)
        self.att = Attention(dim, heads=attention_heads, dim_head=attention_heads_dim)
        self.attdrop = (
            nn.Dropout(attention_drop) if attention_drop > 0.0 else nn.Identity()
        )
        self.conv = conform_conv(
            dim,
            kernel_size=kernel_size,
            DropoutL=conv_drop,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        mask=None,
    ):
        x = self.ffn1(self.norm1(x)) * 0.5 + x

        x = self.attdrop(self.att(self.norm2(x), mask=mask)) + x
        x = self.conv(self.norm3(x)) + x
        x = self.ffn2(self.norm4(x)) * 0.5 + x
        return self.norm5(x)

        # return x


class Gcf(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,
        conv_drop: float = 0.1,
        ffn_latent_drop: float = 0.1,
        ffn_out_drop: float = 0.1,
        attention_drop: float = 0.1,
        attention_heads: int = 4,
        attention_heads_dim: int = 64,
    ):
        super().__init__()
        self.att1 = conform_blocke(
            dim=dim,
            kernel_size=kernel_size,
            conv_drop=conv_drop,
            ffn_latent_drop=ffn_latent_drop,
            ffn_out_drop=ffn_out_drop,
            attention_drop=attention_drop,
            attention_heads=attention_heads,
            attention_heads_dim=attention_heads_dim,
        )
        self.att2 = conform_blocke(
            dim=dim,
            kernel_size=kernel_size,
            conv_drop=conv_drop,
            ffn_latent_drop=ffn_latent_drop,
            ffn_out_drop=ffn_out_drop,
            attention_drop=attention_drop,
            attention_heads=attention_heads,
            attention_heads_dim=attention_heads_dim,
        )
        self.glu1 = nn.Sequential(nn.Linear(dim, dim * 2), GLU(2))
        self.glu2 = nn.Sequential(nn.Linear(dim, dim * 2), GLU(2))

    def forward(self, midi, bound):
        midi = self.att1(midi)
        bound = self.att2(bound)
        midis = self.glu1(midi)
        bounds = self.glu2(bound)
        return midi + bounds, bound + midis


class Gmidi_conform(nn.Module):
    def __init__(
        self,
        lay: int,
        dim: int,
        indim: int,
        outdim: int,
        use_lay_skip: bool,
        kernel_size: int = 31,
        conv_drop: float = 0.1,
        ffn_latent_drop: float = 0.1,
        ffn_out_drop: float = 0.1,
        attention_drop: float = 0.1,
        attention_heads: int = 4,
        attention_heads_dim: int = 64,
    ):
        super().__init__()

        self.inln = nn.Linear(indim, dim)
        self.inln1 = nn.Linear(indim, dim)
        self.outln = nn.Linear(dim, outdim)
        self.cutheard = nn.Linear(dim, 1)
        # self.cutheard = nn.Linear(dim, outdim)
        self.lay = lay
        self.use_lay_skip = use_lay_skip
        self.cf_lay = nn.ModuleList(
            [
                Gcf(
                    dim=dim,
                    kernel_size=kernel_size,
                    conv_drop=conv_drop,
                    ffn_latent_drop=ffn_latent_drop,
                    ffn_out_drop=ffn_out_drop,
                    attention_drop=attention_drop,
                    attention_heads=attention_heads,
                    attention_heads_dim=attention_heads_dim,
                )
                for _ in range(lay)
            ]
        )
        self.att1 = conform_blocke(
            dim=dim,
            kernel_size=kernel_size,
            conv_drop=conv_drop,
            ffn_latent_drop=ffn_latent_drop,
            ffn_out_drop=ffn_out_drop,
            attention_drop=attention_drop,
            attention_heads=attention_heads,
            attention_heads_dim=attention_heads_dim,
        )
        self.att2 = conform_blocke(
            dim=dim,
            kernel_size=kernel_size,
            conv_drop=conv_drop,
            ffn_latent_drop=ffn_latent_drop,
            ffn_out_drop=ffn_out_drop,
            attention_drop=attention_drop,
            attention_heads=attention_heads,
            attention_heads_dim=attention_heads_dim,
        )

    def forward(self, x, mask=None):
        x1 = x.clone()

        x = self.inln(x)
        x1 = self.inln1(x1)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        for idx, i in enumerate(self.cf_lay):
            x, x1 = i(x, x1)

            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x, x1 = self.att1(x), self.att2(x1)

        cutprp = self.cutheard(x1)
        midiout = self.outln(x)

        return midiout, cutprp
