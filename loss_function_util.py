import torch.nn as nn
import torch
import math


class GaborSTRFConv(nn.Module):

    """Gabor-STRF-based cross-correlation kernel."""

    def __init__(self, supn, supk, nkern, rates=None, scales=None, norm_strf=True, use_hz=False, strf_type='real', activation="None", real_time=False, separable=False, groups=1, stride=[1, 1], requires_grad=False,clamp_param=False):
        """Instantiate a Gabor-based STRF convolution layer.
        Parameters
        ----------
        supn: int
            Time support in number of frames. Also the window length.
        supk: int
            Frequency support in number of channels. Also the window length.
        nkern: int
            Number of kernels, each with a learnable rate and scale.
        rates: list of float, None
            Initial values for temporal modulation.
        scales: list of float, None
            Initial values for spectral modulation.
        """
        super(GaborSTRFConv, self).__init__()
        self.numN = supn
        self.numK = supk
        self.numKern = nkern
        self.use_hz = use_hz
        self.norm_strf = norm_strf
        self.real_time = real_time
        self.separable = separable
        self.groups = groups
        self.stride = stride
        self.strf_type = strf_type
        self.activation = activation
        self.clamp_param = clamp_param

        if strf_type == 'both':
            nkern = nkern//2

        assert strf_type in ['both', 'mag',
                             'real'], '[{}]  invalid strf type'.format(strf_type)
        assert activation in [
            'None', 'abs', 'relu'], '[{}] invalid activation'.format(activation)

        if supk % 2 == 0:  # force odd number
            supk += 1
        self.supk = torch.arange(supk, dtype=torch.float32)
        if supn % 2 == 0:  # force odd number
            supn += 1
        self.supn = torch.arange(supn, dtype=self.supk.dtype)
        self.padding = (supn//2, supk//2)
        # Set up learnable parameters
        # for param in (rates, scales):
        #    assert (not param) or len(param) == nkern
        if self.activation == 'relu':
            self.activation = nn.ReLU()
        elif self.activation == 'abs':
            self.activation = torch.abs
        elif self.activation == 'None':
            self.activation = None
        else:
            raise NotImplementedError(
                "{} activation not implemented".format(self.activation))

        if rates is None:

            if self.use_hz:
                rates = torch.rand(nkern) * 25
            else:
                rates = torch.rand(nkern) * math.pi/2

        if scales is None:

            if self.use_hz:
                scales = (torch.rand(nkern) * 2.0 - 1.0)/2.0
            else:
                scales = (torch.rand(nkern)*2.0-1.0) * math.pi/2.0

        self.rates_ = nn.Parameter(torch.Tensor(
            rates), requires_grad=requires_grad)
        self.scales_ = nn.Parameter(torch.Tensor(
            scales), requires_grad=requires_grad)

    def strfs(self):
        """Make STRFs using the current parameters."""

        if self.supn.device != self.rates_.device:  # for first run
            self.supn = self.supn.to(self.rates_.device)
            self.supk = self.supk.to(self.rates_.device)
        n0, k0 = self.padding

        nwind = .5 - .5 * \
            torch.cos(2*math.pi*(self.supn+1)/(len(self.supn)+1))
        kwind = .5 - .5 * \
            torch.cos(2*math.pi*(self.supk+1)/(len(self.supk)+1))

        new_wind = torch.matmul((nwind).unsqueeze(-1),
                                (kwind).unsqueeze(0))

        if self.separable:

            nsin = torch.sin(torch.ger(self.rates_, self.supn-n0))
            ncos = torch.cos(torch.ger(self.rates_, self.supn-n0))
            ksin = torch.sin(torch.ger(self.scales_, self.supk-k0))
            kcos = torch.cos(torch.ger(self.scales_, self.supk-k0))

            real_strf = torch.bmm((ncos).unsqueeze(-1),
                                  (kcos).unsqueeze(1))

            imag_strf = torch.bmm((nsin).unsqueeze(-1),
                                  (ksin).unsqueeze(1))

        else:

            if self.clamp_param:
                scales = torch.clamp(self.scales_,-torch.pi,torch.pi)
                rates = torch.abs(self.rates_)
            else:
                scales = self.scales_
                rates = self.rates_

            n_n_0 = self.supn - n0
            k_k_0 = self.supk - k0
            n_mult = torch.matmul(n_n_0.unsqueeze(1), torch.ones(
                (1, len(self.supk))).type(torch.FloatTensor).to(self.rates_.device))
            k_mult = torch.matmul(torch.ones((len(self.supn), 1)).type(
                torch.FloatTensor).to(self.rates_.device), k_k_0.unsqueeze(0))

            inside = rates.unsqueeze(1).unsqueeze(
                1)*n_mult + scales.unsqueeze(1).unsqueeze(1)*k_mult
            real_strf = torch.cos(inside)
            imag_strf = torch.sin(inside)

        if self.strf_type == 'real':
            final_strf = real_strf * new_wind.unsqueeze(0)

        else:
            real_strf *= new_wind.unsqueeze(0)
            imag_strf *= new_wind.unsqueeze(0)
            if self.strf_type == 'both':
                final_strf = torch.cat([real_strf, imag_strf], dim=0)
            elif self.strf_type == 'mag':
                final_strf = (real_strf**2 + imag_strf**2)**.5

            else:
                raise NotImplementedError(
                    "{} strf type not implemented".format(self.strf_type))

        if self.norm_strf:
            final_strf = final_strf / (torch.sum(final_strf**2, dim=(1, 2)
                                                 ).unsqueeze(1).unsqueeze(2))**.5

        return final_strf

    def forward(self, sigspec):
        """Forward pass a batch of (real) spectra [Batch x Time x Frequency]."""
        if len(sigspec.shape) == 2:  # expand batch dimension if single eg
            sigspec = sigspec.unsqueeze(0)

        if self.groups == 1:
            sigspec = sigspec.unsqueeze(1)

        strfs = self.strfs().unsqueeze(1).type_as(sigspec)
        out = F.conv2d(sigspec, strfs, stride=self.stride,
                       padding=self.padding, groups=self.groups)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def __repr__(self):
        """Gabor filter"""
        report = """
            +++++ Gabor Filter Kernels [{}], supn[{}], supk[{}] strf_type [{}] activation [{}]use_hz[{}] norm strf [{}] real time [{}] separable [{}] groups [{}] stride [{}] clamp [{}]+++++

        """.format(self.numKern, self.numN, self.numK, self.strf_type, self.activation, self.use_hz, self.norm_strf, self.real_time, self.separable, self.groups, self.stride,self.clamp_param
                   )

        return report

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2, squeeze_type='sigmoid'):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = [
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        ]

        self.squeeze_type = squeeze_type
        if self.squeeze_type == 'sigmoid':
            print('using sigmoid squeeze')
            self.fc.append(nn.Sigmoid())

        elif self.squeeze_type == 'softmax':
            print('using softmax squeeze')
            self.fc.append(nn.Softmax(dim=1))

        self.fc = nn.Sequential(*self.fc)

    def forward(self, x, enhanced=None):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        if enhanced is not None:
            return x * y.expand_as(x), enhanced * y.expand_as(x)

        else:
            return x * y.expand_as(x)