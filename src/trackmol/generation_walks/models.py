import torchvision

print('Torchvision version:', torchvision.__version__)
import torch.nn.functional as F

from tqdm import tqdm
import torch
from torch import nn

print("loaded models\n\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Linear_Variance_Scheduler:
    def __init__(self, time_steps, beta_start, beta_end, device=device):
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.betas = torch.linspace(self.beta_start, self.beta_end, self.time_steps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def diffusion_process(self, x, noise, t):
        sqrt_alpha_bar = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

    def ddpm_sampling(self, model: object, num_samples: object, channels: object, traj_length: object, labels: object,
                      x: object = None, iss: object = None, plot_frequency=100) -> object:  #
        model.eval()
        with torch.inference_mode():
            if x is None:
                x = torch.randn((num_samples, channels, traj_length)).to(self.device)
            assert x.shape == (num_samples, channels,
                               traj_length), f"Expected x to have shape {(num_samples, channels, traj_length)}, but got {x.shape}"
            collect = []
            for i in tqdm(reversed(range(self.time_steps))):
                t = (torch.ones(num_samples) * i).long().to(self.device)
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
                x = x.to(self.device)
                labels = labels.to(self.device)
                pred_noise = model(x.squeeze(), t, labels)
                alphas = self.alphas[t][:, None, None]
                alpha_bar = self.alpha_bar[t][:, None, None]
                betas = self.betas[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x) if iss is None else iss[i - 1]
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alphas) * (
                        x - ((1 - alphas) / (torch.sqrt(1 - alpha_bar))) * pred_noise) + torch.sqrt(betas) * noise
                if (i + 1) % plot_frequency == 0 or i == 0:
                    collect.append(x)
        return x, collect


"""# Transformer"""


class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model

        # Linear projection for input trajectories
        self.trajectory_encoder = nn.Linear(2, d_model)

        # Linear projection for context (flattened to d_model)
        self.context_encoder = nn.Linear(17, d_model)

        # Transformer layers
        self.trajectory_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )

        self.cross_attention = nn.MultiheadAttention(d_model, nhead)

        # Transformer decoder to generate output trajectory
        self.output_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )

        self.output_projection = nn.Linear(d_model, 2)

    def forward(self, trajectory, t, embedding):
        """
        trajectory: (batch_size, 32, 2) or (batch_size, 2, 32), batch size may be omitted for single-traj datasets
        context: (batch_size, 17) or (batch_size, 17, 1)
        """
        if trajectory.dim() == 4:
            trajectory = trajectory.squeeze(1)
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
        if trajectory.shape[-1] == 32:
            trajectory = trajectory.permute(0, 2, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        context = torch.concat([embedding, t], dim=1)
        # Encode trajectory
        traj_emb = self.trajectory_encoder(trajectory)  # (batch, 32, d_model)

        # Encode context (flattened to d_model dimension)
        if context.dim() == 3:  # Handle optional extra dimension
            context = context.squeeze(-1)
        ctx_emb = self.context_encoder(context).unsqueeze(1)  # (batch, 1, d_model)

        # Process trajectory through transformer
        traj_enc = self.trajectory_transformer(traj_emb.permute(1, 0, 2))  # (32, batch, d_model)
        # Cross-attention: Trajectory attends to the single context embedding
        traj_cross, _ = self.cross_attention(traj_enc, ctx_emb.permute(1, 0, 2), ctx_emb.permute(1, 0, 2))

        # Decode the trajectory using transformer decoder
        traj_out = self.output_transformer(traj_cross)  # (32, batch, d_model)

        # Project back to 2D space
        output = self.output_projection(traj_out.permute(1, 0, 2))  # (batch, 32, 2)
        output = output.permute(0, 2, 1)
        return output


"""# Conditional UNET 1"""


class ResBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, mid_ch=None, residual=False):
        super(ResBlock, self).__init__()

        self.in_channels = inp_ch

        self.residual = residual
        if not mid_ch:
            mid_ch = out_ch
        self.resnet_conv = nn.Sequential()
        self.resnet_conv.add_module('conv1', nn.Conv1d(in_channels=inp_ch, out_channels=mid_ch, kernel_size=3, stride=1,
                                                       padding=1))
        self.resnet_conv.add_module('gnor1', nn.GroupNorm(num_groups=8, num_channels=mid_ch))
        self.resnet_conv.add_module('silu1', nn.SiLU())
        self.resnet_conv.add_module('conv2', nn.Conv1d(in_channels=mid_ch, out_channels=out_ch, kernel_size=3, stride=1,
                                                       padding=1))
        self.resnet_conv.add_module('gnor2', nn.GroupNorm(num_groups=8, num_channels=out_ch))

    def forward(self, x):
        if self.residual:
            return x + self.resnet_conv(x)
        else:
            return self.resnet_conv(x)


# SelfAttentionBlock (Modified for 1D input)
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionBlock, self).__init__()

        self.attn_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)

    def forward(self, x):
        b, c, t = x.shape
        inp_attn = x.reshape(b, c, t)
        inp_attn = self.attn_norm(inp_attn)
        inp_attn = inp_attn.transpose(1, 2)
        out_attn, _ = self.mha(inp_attn, inp_attn, inp_attn)
        out_attn = out_attn.transpose(1, 2).reshape(b, c, t)
        return x + out_attn


class DownBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, t_emb_dim=16):
        super(DownBlock, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            ResBlock(inp_ch=inp_ch, out_ch=inp_ch, residual=True),
            ResBlock(inp_ch=inp_ch, out_ch=out_ch)
        )

        self.t_emb_layers = nn.Sequential()
        self.t_emb_layers.add_module('silu1', nn.SiLU())
        self.t_emb_layers.add_module('linr1', nn.Linear(in_features=t_emb_dim, out_features=out_ch))

    def forward(self, x, t):
        # Assert input x has the correct shape [batch_size, inp_ch, seq_len]
        assert x.dim() == 3, f"Expected x to be a 3D tensor, but got {x.shape}"
        assert x.shape[1] == self.down[
            2].in_channels, f"Expected x channels to be {self.down[2].in_channels}, but got {x.shape[1]}"

        # Pass x through the down block
        x = self.down(x)

        # Assert t has the correct shape [batch_size, t_emb_dim]
        assert t.dim() == 2, f"Expected t to be a 2D tensor, but got {t.shape}"
        assert t.shape[1] == self.t_emb_layers[
            1].in_features, f"Expected t features to be {self.t_emb_layers[1].in_features}, but got {t.shape[1]}"
        # Pass t through the t_emb_layers
        t_emb = self.t_emb_layers(t)[:, :, None].repeat(1, 1, x.shape[2])

        # Assert the final shape of t_emb matches x
        assert t_emb.shape[0] == x.shape[
            0], f"Expected batch size of t_emb to match x, but got {t_emb.shape[0]} vs {x.shape[0]}"
        assert t_emb.shape[1] == x.shape[
            1], f"Expected channel size of t_emb to match x, but got {t_emb.shape[1]} vs {x.shape[1]}"
        assert t_emb.shape[2] == x.shape[
            2], f"Expected sequence length of t_emb to match x, but got {t_emb.shape[2]} vs {x.shape[2]}"

        return x + t_emb


class UpBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, t_emb_dim=16):
        super(UpBlock, self).__init__()

        self.upsamp = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.up = nn.Sequential(
            ResBlock(inp_ch=inp_ch, out_ch=inp_ch, residual=True),
            ResBlock(inp_ch=inp_ch, out_ch=out_ch, mid_ch=inp_ch // 2)
        )

        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_ch)
        )

    def forward(self, x, skip, t):
        x = self.upsamp(x)
        x = torch.cat([skip, x], dim=1)
        x = self.up(x)
        t_emb = self.t_emb_layers(t)[:, :, None].repeat(1, 1, x.shape[2])
        return x + t_emb


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, context_dim, nheads=4, ngroups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=ngroups, num_channels=channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(context_dim, channels)
        self.v_proj = nn.Linear(context_dim, channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=nheads, batch_first=True)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x, context):
        # x: [B, C, T] -> [B, T, C]
        x_ = self.norm(x).permute(0, 2, 1)
        q = self.q_proj(x_)

        # context: [B, C_ctx] or [B, T_ctx, C_ctx] -> project to k, v
        if context.dim() == 2:
            context = context.unsqueeze(1)  # [B, 1, C_ctx]
        k = self.k_proj(context)
        v = self.v_proj(context)

        out, _ = self.attn(q, k, v)
        out = self.out_proj(out)

        # Back to [B, C, T]
        out = out.permute(0, 2, 1)
        return x + out


class Conditional_UNet(nn.Module):
    def __init__(self, t_emb_dim, device='cuda'):
        super(Conditional_UNet, self).__init__()

        self.device = device
        self.t_emb_dim = t_emb_dim

        # Define network layers
        self.inp = ResBlock(inp_ch=2, out_ch=64, residual=False)  # Consider residual=False at the first block
        self.down1 = DownBlock(inp_ch=64, out_ch=128, t_emb_dim=t_emb_dim)
        self.sa1 = SelfAttentionBlock(channels=128)
        self.down2 = DownBlock(inp_ch=128, out_ch=256, t_emb_dim=t_emb_dim)
        self.sa2 = SelfAttentionBlock(channels=256)
        self.down3 = DownBlock(inp_ch=256, out_ch=256, t_emb_dim=t_emb_dim)
        self.sa3 = SelfAttentionBlock(channels=256)

        self.lat1 = ResBlock(inp_ch=256, out_ch=512)
        self.lat2 = ResBlock(inp_ch=512, out_ch=512)
        self.lat3 = ResBlock(inp_ch=512, out_ch=256)

        self.up1 = UpBlock(inp_ch=512, out_ch=128, t_emb_dim=t_emb_dim)
        self.sa4 = SelfAttentionBlock(channels=128)
        self.up2 = UpBlock(inp_ch=256, out_ch=64, t_emb_dim=t_emb_dim)
        self.sa5 = SelfAttentionBlock(channels=64)
        self.up3 = UpBlock(inp_ch=128, out_ch=64, t_emb_dim=t_emb_dim)
        self.sa6 = SelfAttentionBlock(channels=64)

        self.out = nn.Conv1d(in_channels=64, out_channels=2, kernel_size=1)  # 2 for x, y coordinates

        # Add a linear layer to project conditioning labels to the same dimension as time embeddings
        self.conditioning_projection = nn.Linear(16, t_emb_dim)  # Project the conditioning labels to t_emb_dim

    def position_embeddings(self, t, channels):
        i = 1 / (10000 ** (torch.arange(start=0, end=channels, step=2, device=self.device) / channels))
        pos_emb_sin = torch.sin(t.repeat(1, channels // 2) * i)
        pos_emb_cos = torch.cos(t.repeat(1, channels // 2) * i)
        pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)
        return pos_emb

    def forward(self, x, t, conditioning_labels):
        # Ensure t is reshaped correctly
        t = t.unsqueeze(1).float()  # Shape (batch_size, 1)
        t_emb = self.position_embeddings(t, self.t_emb_dim)  # Shape (batch_size, t_emb_dim)

        # Project the conditioning labels to the same dimensionality as the time embedding
        conditioned_emb = self.conditioning_projection(conditioning_labels)  # Shape (batch_size, t_emb_dim)

        # Add the projected conditioning labels to the time embeddings
        t_emb += conditioned_emb  # Now t_emb is (batch_size, t_emb_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        assert t_emb.shape == (x.shape[0],
                               self.t_emb_dim), f"Expected t_emb to be of shape {(x.shape[0], self.t_emb_dim)}, but got {t_emb.shape}"
        # Now proceed with the rest of the forward pass
        x = x.squeeze()
        # x = x.permute(0, 2, 1)  # Reshape to (batch_size, channels, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x1 = self.inp(x)  # ResBlock with input shape (batch_size, 2, 32)
        x2 = self.down1(x1, t_emb)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t_emb)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t_emb)
        x4 = self.sa3(x4)

        x4 = self.lat1(x4)
        x4 = self.lat2(x4)
        x4 = self.lat3(x4)

        x = self.up1(x4, x3, t_emb)
        x = self.sa4(x)
        x = self.up2(x, x2, t_emb)
        x = self.sa5(x)
        x = self.up3(x, x1, t_emb)
        x = self.sa6(x)
        output = self.out(x)  # Output shape (batch_size, 2, 32)
        return output


# ======== WITH CROSS ATTENTION ========

class Conditional_UNet_cross_attention(nn.Module):
    def __init__(
            self,
            t_emb_dim,
            base_channels=64,
            channel_mults=(1, 2, 4),
            latent_dim=512,
            device='cuda',
            ngroups=8,
            nheads=4,
    ):
        super(Conditional_UNet_cross_attention, self).__init__()

        self.device = device
        self.t_emb_dim = t_emb_dim

        # Calculate channel sizes
        ch1 = base_channels * channel_mults[0]  # 64
        ch2 = base_channels * channel_mults[1]  # 128
        ch3 = base_channels * channel_mults[2]  # 256

        # Input block
        self.inp = ResBlock(inp_ch=2, out_ch=ch1, residual=False, ngroups=ngroups)

        # Downsampling
        self.down1 = DownBlock(inp_ch=ch1, out_ch=ch2, t_emb_dim=t_emb_dim, ngroups=ngroups)
        self.sa1 = CrossAttentionBlock(channels=ch2, context_dim=t_emb_dim, nheads=nheads)
        self.down2 = DownBlock(inp_ch=ch2, out_ch=ch3, t_emb_dim=t_emb_dim, ngroups=ngroups)
        self.sa2 = CrossAttentionBlock(channels=ch3, context_dim=t_emb_dim, nheads=nheads)
        self.down3 = DownBlock(inp_ch=ch3, out_ch=ch3, t_emb_dim=t_emb_dim, ngroups=ngroups)
        self.sa3 = CrossAttentionBlock(channels=ch3, context_dim=t_emb_dim, nheads=nheads)

        # Latent blocks
        self.lat1 = ResBlock(inp_ch=ch3, out_ch=latent_dim, ngroups=ngroups)
        self.lat2 = ResBlock(inp_ch=latent_dim, out_ch=latent_dim, ngroups=ngroups)
        self.lat3 = ResBlock(inp_ch=latent_dim, out_ch=ch3, ngroups=ngroups)

        # Upsampling
        self.up1 = UpBlock(inp_ch=ch3 + ch3, out_ch=ch2, t_emb_dim=t_emb_dim, ngroups=ngroups)
        self.sa4 = CrossAttentionBlock(channels=ch2, context_dim=t_emb_dim, nheads=nheads)
        self.up2 = UpBlock(inp_ch=ch2 + ch2, out_ch=ch1, t_emb_dim=t_emb_dim, ngroups=ngroups)
        self.sa5 = CrossAttentionBlock(channels=ch1, context_dim=t_emb_dim, nheads=nheads)
        self.up3 = UpBlock(inp_ch=ch1 + ch1, out_ch=ch1, t_emb_dim=t_emb_dim, ngroups=ngroups)
        self.sa6 = CrossAttentionBlock(channels=ch1, context_dim=t_emb_dim, nheads=nheads)

        self.out = nn.Conv1d(in_channels=ch1, out_channels=2, kernel_size=1)

        # Project conditioning labels to time embedding dimension
        self.conditioning_projection = nn.Linear(16, t_emb_dim)
        self.context_projection = nn.Linear(17, t_emb_dim)

    def position_embeddings(self, t, channels):
        i = 1 / (10000 ** (torch.arange(start=0, end=channels, step=2, device=self.device) / channels))
        pos_emb_sin = torch.sin(t.repeat(1, channels // 2) * i)
        pos_emb_cos = torch.cos(t.repeat(1, channels // 2) * i)
        pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)
        return pos_emb

    def forward(self, x, t, conditioning_labels):
        t = t.unsqueeze(1).float()

        ctx = torch.concat((t, conditioning_labels), dim=1)
        t_emb = self.context_projection(ctx)

        x = x.squeeze()
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x1 = self.inp(x)
        x2 = self.down1(x1, t_emb)
        x2 = self.sa1(x2, t_emb)
        x3 = self.down2(x2, t_emb)
        x3 = self.sa2(x3, t_emb)
        x4 = self.down3(x3, t_emb)
        x4 = self.sa3(x4, t_emb)

        x4 = self.lat1(x4)
        x4 = self.lat2(x4)
        x4 = self.lat3(x4)

        x = self.up1(x4, x3, t_emb)
        x = self.sa4(x, t_emb)
        x = self.up2(x, x2, t_emb)
        x = self.sa5(x, t_emb)
        x = self.up3(x, x1, t_emb)
        x = self.sa6(x, t_emb)
        output = self.out(x)
        return output


"""## Visualise architecture"""

batch1 = torch.randn((64, 2, 32)).to(device)

ddpm = Linear_Variance_Scheduler(time_steps=1000, beta_start=0.0001, beta_end=0.02, device=device)

"""# Simple model"""


class DenoiseTrajectoryNet(nn.Module):
    def __init__(self):
        super(DenoiseTrajectoryNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(17, 64)
        self.fc2 = nn.Linear(64, 128)

        self.deconv1 = nn.Conv1d(in_channels=64 + 128, out_channels=32, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.deconv3 = nn.Conv1d(in_channels=16, out_channels=2, kernel_size=3, padding=1)  # Output clean trajectory

    def forward(self, noisy_traj, t, labels):
        if noisy_traj.dim() == 4:
            if noisy_traj.size(1) == 1:
                noisy_traj = noisy_traj.squeeze(1)
            elif noisy_traj.size(0) == 1:
                noisy_traj = noisy_traj.squeeze(0)
        if noisy_traj.dim() == 2:
            noisy_traj = noisy_traj.unsqueeze(0)  ###
        if noisy_traj.size(1) != 2:
            noisy_traj = noisy_traj.permute(0, 2, 1)
        assert noisy_traj.dim() == 3 and noisy_traj.size(1) == 2 and noisy_traj.size(
            2) == 32, f"Input tensor must have shap (batch_size, 2, 32), got {noisy_traj.shape} instead."
        batch_size = noisy_traj.size(0)
        if batch_size == 1 and labels.dim() == 1:
            labels = labels.unsqueeze(0)
        assert labels.dim() == 2 and labels.size(0) == batch_size and labels.size(
            1) == 16, f"Labels tensor must have shape ((batch_size = {batch_size}), 16), got {labels.shape} instead."
        assert t.dim() == 1 and t.size(
            0) == batch_size, f"t tensor must have shape (batch_size,), got {t.shape} instead."
        y = torch.concat([labels, t.unsqueeze(1)], dim=1)

        # Extract features from noisy trajectory
        x = F.relu(self.conv1(noisy_traj))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # (batch, 64, 32)

        # Process labels
        y = F.relu(self.fc1(y))  # (batch, 64)
        y = F.relu(self.fc2(y))  # (batch, 128)
        y = y.unsqueeze(2).expand(-1, -1, 32)

        x = torch.cat([x, y], dim=1)

        # Decode to clean trajectory
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        clean_traj = self.deconv3(x)

        return clean_traj.squeeze()


class DenoiseTrajectoryNet_pooling(nn.Module):
    def __init__(self, conv_outc_1=16, conv_outc_2=32, conv_outc_3=64, conv_outc_4=128, conv_outc_5=256, kernel_size=3,
                 param_space_1=64, param_space_2=128):
        super(DenoiseTrajectoryNet_pooling, self).__init__()

        print("DenoiseTrajectoryNet initialized with parameters:")
        print(
            f"\tconv_outc_1: {conv_outc_1},\n\tconv_outc_2: {conv_outc_2},\n\tconv_outc_3: {conv_outc_3},\n\tconv_outc_4: {conv_outc_4},\n\tconv_outc_5: {conv_outc_5},\n\tkernel_size: {kernel_size},\n\tparam_space_1: {param_space_1},\n\tparam_space_2: {param_space_2}.")

        self.conv1 = nn.Conv1d(2, conv_outc_1, kernel_size, padding=kernel_size // 3)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(conv_outc_1, conv_outc_2, kernel_size, padding=kernel_size // 3)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(conv_outc_2, conv_outc_3, kernel_size, padding=kernel_size // 3)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(conv_outc_3, conv_outc_4, kernel_size, padding=kernel_size // 3)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(conv_outc_4, conv_outc_5, kernel_size, padding=kernel_size // 3)
        self.pool5 = nn.MaxPool1d(2)

        # Fully connected layers
        self.fc1 = nn.Linear(17, param_space_1)
        self.fc2 = nn.Linear(param_space_1, param_space_2)

        # Upsample + deconv layers (decoder)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv5 = nn.Conv1d(conv_outc_5 + param_space_2, conv_outc_4, kernel_size, padding=kernel_size // 3)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv4 = nn.Conv1d(conv_outc_4, conv_outc_3, kernel_size, padding=kernel_size // 3)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv3 = nn.Conv1d(conv_outc_3, conv_outc_2, kernel_size, padding=kernel_size // 3)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv2 = nn.Conv1d(conv_outc_2, conv_outc_1, kernel_size, padding=kernel_size // 3)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1 = nn.Conv1d(conv_outc_1, 2, kernel_size, padding=1)

    def forward(self, noisy_traj, t, labels):
        if noisy_traj.dim() == 4:
            if noisy_traj.size(1) == 1:
                noisy_traj = noisy_traj.squeeze(1)
            elif noisy_traj.size(0) == 1:
                noisy_traj = noisy_traj.squeeze(0)
        if noisy_traj.size(1) != 2:
            noisy_traj = noisy_traj.permute(0, 2, 1)
        assert noisy_traj.dim() == 3 and noisy_traj.size(1) == 2 and noisy_traj.size(2) == 32

        batch_size = noisy_traj.size(0)
        if batch_size == 1 and labels.dim() == 1:
            labels = labels.unsqueeze(0)
        assert labels.dim() == 2 and labels.size(0) == batch_size and labels.size(1) == 16
        assert t.dim() == 1 and t.size(0) == batch_size
        y = torch.concat([labels, t.unsqueeze(1)], dim=1)

        x = F.relu(self.conv1(noisy_traj))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        # Label embedding
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = y.unsqueeze(2).expand(-1, -1, 1)

        x = torch.cat([x, y], dim=1)

        x = self.upsample5(x)
        x = F.relu(self.deconv5(x))

        x = self.upsample4(x)
        x = F.relu(self.deconv4(x))

        x = self.upsample3(x)
        x = F.relu(self.deconv3(x))

        x = self.upsample2(x)
        x = F.relu(self.deconv2(x))

        x = self.upsample1(x)
        clean_traj = self.deconv1(x)

        return clean_traj.squeeze()
