import torch
import torchaudio
import torch.nn as nn
from pdb import set_trace
import math
import torch.nn.functional as F
from speechbrain.lobes.features import Fbank
from torch.nn.functional import unfold, pad
from speechbrain.processing.features import InputNormalization
from speechbrain.nnet.normalization import LayerNorm
from loss_function_util import GaborSTRFConv, SELayer



class ModulationReconstructDomainLossModule(torch.nn.Module):
    """Modulation-domain loss function developed in [1] for supervised speech enhancement

        In our paper, we used the gabor-based STRF kernels as the modulation kernels and used the log-mel spectrogram as the input spectrogram representation.
        Specific parameter details are in the paper and in the example below

        Parameters
        ----------
        modulation_kernels: nn.Module
            Differentiable module that transforms a spectrogram representation to the modulation domain

            modulation_domain = modulation_kernels(input_tf_representation)
            Input Spectrogram representation (B, T, F) ---> |(M) modulation_kernels| ---> Modulation Domain(B, M, T', F') 

        norm: boolean
            Normalizes the modulation domain representation to be 0 mean across time

        [1] T. Vuong, Y. Xia, and R. M. Stern, “A modulation-domain lossfor neural-network-based real-time speech enhancement”
            Accepted ICASSP 2021, https://arxiv.org/abs/2102.07330


    """

    def __init__(self, modulation_kernels, nfft=400, hop_length=10, window_length=20, distance='L2', norm=False, use_relu=True, n_mels=80, hparams=None):
        super(ModulationReconstructDomainLossModule, self).__init__()

        self.modulation_kernels = modulation_kernels
        self.hparams = hparams
        self.reconstruct_activation = hparams.st_mod.reconstruct_activation
        self.use_squeeze = hparams.st_mod.use_squeeze
        self.reconstruct_distance = hparams.st_mod.reconstruct_distance
        self.reconstruct_norm = hparams.st_mod.reconstruct_norm
        self.alphas = hparams.st_mod.alphas.split('-')
        self.tune_kernels = hparams.training.lr_multiplier_strf != 0
        if self.tune_kernels:
            print('tuning kernels')
        else:
            print('not tuning kernels')
      
        if self.reconstruct_distance == 'L2':
            self.reconstruct_distance = nn.MSELoss()
        elif self.reconstruct_distance == 'L1':
            self.reconstruct_distance = nn.L1Loss()
        else:
            raise NotImplementedError(
                "{} distance not implemented".format(self.reconstruct_distance))

        if self.reconstruct_activation == 'relu':
            self.reconstruct_activation = nn.ReLU()
        elif self.reconstruct_activation == 'prelu':
            self.reconstruct_activation = nn.PReLU(modulation_kernels.numKern)
        else:
            self.reconstruct_activation = None

        if self.use_squeeze:
            self.squeeze = SELayer(
                modulation_kernels.numKern, reduction=hparams.st_mod.squeeze_fraction, squeeze_type=hparams.st_mod.squeeze_type)
            self.scale_mod_squeeze = hparams.st_mod.scale_mod_squeeze
            print('scaling with squeeze [{}]'.format(self.scale_mod_squeeze))
        else:
            self.squeeze = None


        if self.reconstruct_norm == 'layer':
            self.reconstruct_norm = LayerNorm(input_shape=(
                None, None, n_mels, modulation_kernels.numKern))
        else:
            self.reconstruct_norm = None

        self.reconstruct = nn.Conv2d(
            modulation_kernels.numKern, 1, (3, 3), (1, 1), padding=(1, 1))

        if distance == 'L2':
            self.mse = nn.MSELoss(reduce=False)
        elif distance == 'L1':
            self.mse = nn.L1Loss(reduce=False)

        else:
            raise NotImplementedError(
                "{} distance not implemented".format(distance))

        self.norm = norm
        self.fbank = Fbank(sample_rate=16000, n_fft=nfft,
                           n_mels=n_mels, hop_length=hop_length, win_length=window_length).to('cuda')
        

        self.use_relu = use_relu
        self.window_length = window_length

        print('modulation loss with recon [{}], vad [{}] relu [{}]'.format(
            distance, self.use_vad, self.use_relu))
        print('recon activation [{}] use squeeze [{}] recon norm [{}]'.format(
            self.reconstruct_activation, self.use_squeeze, self.reconstruct_norm))

        print('alpha weights [{}]'.format(self.alphas))
        self.input_norm = InputNormalization(norm_type='sentence')

    def switch_grad(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def forward(self, enhanced_wave, clean_wave):
        """Calculate modulation-domain loss
        Args:
            enhanced_spect (Tensor): spectrogram representation of enhanced signal (B, #frames, #freq_channels).
            clean_spect (Tensor): spectrogram representation of clean ground-truth signal (B, #frames, #freq_channels).
        Returns:
            Tensor: Modulation-domain loss value.
        """

        clean_spect = self.fbank(clean_wave.squeeze(1))
        enhanced_spect = self.fbank(enhanced_wave.squeeze(1))

        self.switch_grad(self.modulation_kernels, False)
        clean_mod = self.modulation_kernels(clean_spect)
        enhanced_mod = self.modulation_kernels(enhanced_spect)

        if self.use_squeeze and self.scale_mod_squeeze:
            # don't let kernels change
            # learn to scale modulation and then mse based?
            # scaling could also be softmax instead of sigmoid?
            self.switch_grad(self.squeeze, False)
            clean_mod, enhanced_mod = self.squeeze(clean_mod, enhanced_mod)

        # norm input, relu input, sequeeze input, reconstruct both? learn squeeze for clean and enhanced or just clean?
        if self.use_relu:
            clean_mod = F.relu(clean_mod)
            enhanced_mod = F.relu(enhanced_mod)

        if self.norm:
            mean_clean_mod = torch.mean(clean_mod, dim=2)
            mean_enhanced_mod = torch.mean(enhanced_mod, dim=2)

            clean_mod = clean_mod - mean_clean_mod.unsqueeze(2)
            enhanced_mod = enhanced_mod - mean_enhanced_mod.unsqueeze(2)

        
        mod_mse_loss = self.mse(enhanced_mod, clean_mod)

        mod_mse_loss = torch.mean(torch.sum(mod_mse_loss, dim=(
            1, 2, 3))/torch.sum((clean_mod)**2, dim=(1, 2, 3)))
        
        if self.tune_kernels:
            self.switch_grad(self.modulation_kernels, True)

        lengths = torch.ones([clean_spect.shape[0]])
        clean_spect = self.input_norm(clean_spect, lengths)
        clean_mod = self.modulation_kernels(clean_spect.detach())

        if self.reconstruct_activation is not None:
            self.switch_grad(self.reconstruct_activation, True)
            clean_mod = self.reconstruct_activation(clean_mod)

        if self.squeeze is not None:
            self.switch_grad(self.squeeze, True)
            clean_mod = self.squeeze(clean_mod)

        self.switch_grad(self.reconstruct, True)
        clean_recon = self.reconstruct(clean_mod).squeeze(1)

        loss_clean = self.reconstruct_distance(clean_recon, clean_spect)

        total_loss = mod_mse_loss * \
            float(self.alphas[0]) + loss_clean * \
            float(self.alphas[1]) 

        return total_loss, mod_mse_loss, loss_clean


    
