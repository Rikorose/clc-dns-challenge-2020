import torch
from torch import Tensor, nn
from torch.jit import Final


class CLCNetStep(nn.Module):
    """CLCNet Module that performes the linear combination."""

    clc_order: Final[int]
    clc_offset: Final[int]
    clc_lookahead: Final[int]
    n_freq_bins: Final[int]
    out_act_f: Final[float]

    def __init__(self, config):
        super().__init__()
        self.eps = config.eps
        self.register_buffer("n_fft", torch.tensor([320]))  # 20 ms at 16 kHz
        self.register_buffer("hop", torch.tensor([config.stft_hop]))
        self.n_freq_bins = self.n_fft.item() // 2 + 1
        config.n_freq_bins = self.n_freq_bins
        self.register_buffer("hann", torch.hann_window(self.n_fft.item()))
        self.register_buffer("hann_sq", self.hann.pow(2))
        self.register_buffer("hann_norm", self.hann_sq.sum().sqrt())
        self.register_buffer("sr", torch.tensor([config.sr]))

        self.norm = ExponentialDecay(alpha=config.norm_alpha)
        self.norm_init_length = config.norm_init_length
        self.db_mult = float(config.norm_db_mult)
        in_features = config.n_freq_bins * 2  # Complex
        self.fc_clc_1 = nn.Linear(in_features, in_features)
        self.bn_clc_1 = nn.BatchNorm1d(in_features)
        self.rnn_clc = nn.GRU(
            in_features,
            config.rnn_n_hidden_clc,
            config.rnn_n_layers_clc,
            bidirectional=False,
            dropout=float(config.rnn_dropout),
            bias=True,
        )
        self.fc_clc_2 = nn.Linear(config.rnn_n_hidden_clc, config.rnn_n_hidden_clc)
        self.bn_clc_2 = nn.BatchNorm1d(config.rnn_n_hidden_clc)
        if config.clc_max_freq > 0:
            self.clc_n_bins = int(floor(config.clc_max_freq / 250))
        else:
            self.clc_n_bins = config.n_freq_bins
        self.fc_clc_out = nn.Linear(
            config.rnn_n_hidden_clc, self.clc_n_bins * config.clc_order * 2, bias=True,
        )
        self.out_act_f = config.out_act_factor
        self.clc_order = config.clc_order
        self.clc_offset = config.clc_offset
        self.clc_lookahead = config.clc_lookahead
        self.n_freq_bins = config.n_freq_bins

    def forward(
        self, x: Tensor, h_norm: Tensor, h_rnn: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward for one time domain sample of size n_fft.

        Args:
            x (Tensor): Frequency domain frame to denoise
            h_norm (Tensor): State of the exponential normalization
            h_rnn (Tensor): State if the GRU layer

        Returns:
            x_enh (Tensor): Enhanced time domain audio signal of input x
            h_norm (Tensor): New norm state
            h_rnn (Tensor): New GRU layer state
        """
        t = 1
        b = 1
        # Only take the current frame as input for the RNN.
        # The context needs to be included, since the CLC coefs will be applied on it.
        # In this case we have order=5, offset=0 and lookahead=0. Thus, x needs to have
        # 5 frames context in TF domain and only the last frame is fed into the RNN.
        x_w = x[-1]
        # Transform complex to magnitude
        x_m = torch.sqrt(x.pow(2).sum(-1) + self.eps)
        # Compute the exponential decay
        x_m, h_norm = self.norm(x_m, h_norm)
        # Unit normalization using resulting magnitude x_m
        rnn_in = x_w.div(x_m.add(self.eps).unsqueeze_(-1))
        rnn_in = rnn_in.view(t, b, -1).div(
            50
        )  # This factor was just to shift it to a better scale
        # FC1 BN ReLU
        rnn_in = self.fc_clc_1(rnn_in)
        rnn_in = self.bn_clc_1(rnn_in.transpose(1, 2)).transpose(1, 2)
        rnn_in = torch.relu(rnn_in)
        # 2 Layer GRU
        clc_coef, h_rnn = self.rnn_clc(rnn_in, h_rnn)
        # FC2 BN ReLU
        clc_coef = self.fc_clc_2(clc_coef)
        clc_coef = self.bn_clc_2(clc_coef.transpose(1, 2)).transpose(1, 2)
        clc_coef = torch.relu(clc_coef)
        # Output FC with tanh
        clc_coef = torch.tanh(self.fc_clc_out(clc_coef)) * self.out_act_f
        clc_coef = clc_coef.view(t, b, self.clc_order, self.clc_n_bins, 2).transpose(
            1, 2
        )
        if self.clc_lookahead > 0:
            clc_coef = clc_coef[self.clc_lookahead :]
            x = x[: -self.clc_lookahead]
        # Complex multiplication of the predicted coefs and the noisy spectrum
        out = complex_mul(x, clc_coef.squeeze()).sum(dim=0)
        return out, h_norm, h_rnn

    def fft_step(self, x: Tensor, clc_tmp: Tensor) -> Tensor:
        clc_tmp = clc_tmp.roll(-1, dims=(0,))
        clc_tmp[-1] = (
            torch.rfft(x * self.hann, signal_ndim=1, normalized=False) / self.hann_norm
        )
        return clc_tmp

    def ifft_step(
        self, fframe: Tensor, tframe: Tensor, buf_wnorm: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Inverse transform with overlap add
        tframe.add_(
            torch.irfft(
                fframe * self.hann_norm,
                signal_ndim=1,
                normalized=False,
                signal_sizes=(self.n_fft.item(),),
            ).mul_(self.hann)
        )
        buf_wnorm = buf_wnorm.roll(-self.hop.item(), dims=(0,))
        buf_wnorm[-self.hop.item() :] = 0.0
        buf_wnorm += self.hann_sq
        norm = torch.clamp_min(buf_wnorm[: self.hop.item()], 1e-10)
        # norm = buf_wnorm[: self.hop.item()]
        # norm = torch.where(norm > 1e-10, norm, torch.ones_like(norm))
        tframe[: self.hop.item()].div_(norm)
        return buf_wnorm

    def init_buffers(self):
        buf_norm = torch.zeros(self.n_freq_bins)
        buf_rnn = torch.zeros(self.rnn_clc.num_layers, 1, self.rnn_clc.hidden_size)
        buf_clc = torch.zeros(self.clc_order, self.n_freq_bins, 2)
        buf_ola_wnorm = torch.zeros(self.n_fft.item())
        return buf_norm, buf_rnn, buf_clc, buf_ola_wnorm


class ExponentialUpdate(nn.Module):
    alpha: Final[int]

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: Tensor, state: Tensor) -> Tensor:
        return x * (1 - self.alpha) + state * self.alpha


class ExponentialDecay(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.update_rule = ExponentialUpdate(alpha)

    def forward(self, x: Tensor, state: Optional[Tensor] = None):
        out = torch.empty_like(x)
        if state is None:
            state = x[0]
        for t in range(x.shape[0]):
            state = self.update_rule(x[t], state)
            out[t] = state
        return out, state


def complex_mul(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """ complex multiplication: a * b = (x + iy) * (u + iv)
            = xu + iyu + xiv - yv

    Args:
        a (Tensor): Nominator of shape [..., 2], where a[..., 0] are all real values and
            a[..., 1] are all imaginary values.
        b (Tensor): Denominator of the same shape as a.
        out (Optional[Tensor]): Already allocated output Tensor, which will be returned.
    """
    if out is None:
        out = torch.empty_like(a)
    out[..., 0] = a[..., 0] * b[..., 0]  # re1
    out[..., 0] -= a[..., 1] * b[..., 1]  # re2
    out[..., 1] = a[..., 0] * b[..., 1]  # im1
    out[..., 1] += a[..., 1] * b[..., 0]  # im2
    return out


def export_jit_frame(model: CLCNetStep, export_file):
    model.eval()
    # Test forward path and generate intermediate dummy output
    frame_size = model.n_fft.item()
    dummy_frame = torch.empty((frame_size,), dtype=torch.float32).uniform_(-1, 1)
    dummy_frame_out = torch.zeros_like(dummy_frame)
    buf_norm, buf_rnn, buf_clc, buf_ola_wnorm = model.init_buffers()
    buf_clc = model.fft_step(dummy_frame, buf_clc)
    dummy_out, buf_norm, buf_rnn = model(buf_clc, buf_norm, buf_rnn)
    buf_ola_wnorm = model.ifft_step(dummy_out, dummy_frame_out, buf_ola_wnorm)
    traced_module = torch.jit.trace_module(
        model,
        inputs={
            "forward": (buf_clc, buf_norm, buf_rnn),
            "fft_step": (dummy_frame, buf_clc),
            "ifft_step": (dummy_out, dummy_frame_out, buf_ola_wnorm),
            "init_buffers": tuple(),
        },
    )
    traced_module.save(export_file)
