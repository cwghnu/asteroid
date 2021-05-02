import json
import os
import torch
from torch import nn
from sklearn.cluster import KMeans
from functools import partial

from asteroid import torch_utils
import asteroid_filterbanks as fb
from asteroid.engine.optimizers import make_optimizer
from asteroid_filterbanks.transforms import mag, apply_mag_mask
from asteroid.dsp.vad import ebased_vad
from asteroid.masknn.recurrent import SingleRNN
from asteroid.utils.torch_utils import pad_x_to_y


def stft_torch(y, n_fft, hop_length):
    assert y.dim() == 2

    # window = th.hann_window(n_fft).to(device)
    # return th.stft(y, n_fft, hop_length, win_length, window=window, return_complex=False)
    return torch.stft(y, n_fft, hop_length, return_complex=False)


def istft_torch(complex_tensor, n_fft, hop_length, length=None, use_mag_phase=False):
    # window = th.hann_window(n_fft).to(device)

    if use_mag_phase:
        assert isinstance(complex_tensor, tuple) or isinstance(complex_tensor, list)
        mag, phase = complex_tensor
        complex_tensor = torch.stack([(mag * torch.cos(phase)), (mag * torch.sin(phase))], dim=-1)

    # return th.istft(complex_tensor, n_fft, hop_length, win_length, window, length=length)
    return torch.istft(complex_tensor, n_fft, hop_length, length=length)


def mag_phase(complex_tensor):
    mag = (complex_tensor.pow(2.0).sum(-1) + 1e-8).pow(0.5 * 1.0)
    phase = torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])
    return mag, phase


class IPDFeature(nn.Module):
    """
    Compute inter-channel phase difference
    """

    def __init__(self, n_channel=2, cos=True, sin=False):
        super(IPDFeature, self).__init__()

        # ipd_index="1,0;2,0;3,0;4,0;5,0;6,0"

        assert n_channel > 1

        for idx in range(n_channel-1):
            if idx == 0:
                ipd_index = "{},{}".format(idx + 1, 0)
            else:
                ipd_index = "{};{},{}".format(ipd_index, idx + 1, 0)

        split_index = lambda sstr: [tuple(map(int, p.split(","))) for p in sstr.split(";")]
        # ipd index
        pair = split_index(ipd_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.ipd_index = ipd_index
        self.cos = cos
        self.sin = sin
        self.num_pairs = len(pair) * 2 if cos and sin else len(pair)

    def extra_repr(self):
        return f"ipd_index={self.ipd_index}, cos={self.cos}, sin={self.sin}"

    def forward(self, p):
        """
        Accept multi-channel phase and output inter-channel phase difference
        args
            p: phase matrix, N x C x F x T
        return
            ipd: N x MF x T
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(self.__name__, p.dim())
            )
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        N, _, _, T = p.shape
        pha_dif = p[:, self.index_l, ...] - p[:, self.index_r, ...]
        if self.cos:
            # N x M x F x T
            ipd = torch.cos(pha_dif)
            if self.sin:
                # N x M x 2F x T
                ipd = torch.cat([ipd, torch.sin(pha_dif)], 2)
        else:
            ipd = torch.fmod(pha_dif, 2 * math.pi) - math.pi
        # N x MF x T
        ipd = ipd.view(N, -1, T)
        # N x MF x T
        return ipd


class RSAN_module(nn.Module):
    def __init__(
        self,
        in_spat=257,
        n_bin=257,
        n_layers=2,
        hidden_size=600,
        dropout=0,
        bidirectional=True,
        spk_emb_dim=128,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            n_bin * 2 + in_spat,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

        self.spk_emb_dim = spk_emb_dim

        self.fc_lstm = nn.Sequential(nn.Linear(hidden_size * 2, spk_emb_dim), nn.Sigmoid())

        self.fc_adpt = nn.Sequential(nn.Linear(spk_emb_dim, spk_emb_dim), nn.Sigmoid())

        self.fc_mask = nn.Sequential(nn.Linear(spk_emb_dim, n_bin), nn.Sigmoid())

        self.fc_gate = nn.Sequential(nn.Linear(spk_emb_dim, spk_emb_dim), nn.Sigmoid())

        self.fc_emb = nn.Sequential(
            nn.Linear(spk_emb_dim, 50), nn.Linear(50, 50), nn.Linear(50, spk_emb_dim), nn.Sigmoid()
        )

        self.fc_affine = nn.Linear(spk_emb_dim, spk_emb_dim, bias=False)

    def forward(self, spec, spat, res_mask, spk_info):
        # spec: [N, F, T]
        # spat: [N, F, T]
        # res_mask: [N, F, T]
        # spk_info: [spk_emb_dim]

        N, F, T = spec.shape

        spec = spec.permute(0, 2, 1).contiguous()
        spat = spat.permute(0, 2, 1).contiguous()
        res_mask = res_mask.permute(0, 2, 1).contiguous()

        input_lstm = torch.cat([spec, spat, res_mask], axis=2)  # [N, T, F*3]

        output_lstm, _ = self.lstm(input_lstm)  # [N, T, hidden_size*2]
        output_lstm = self.fc_lstm(output_lstm)  # [N, T, spk_emb_dim]

        speech_info = self.fc_adpt(spk_info)  # [spk_emb_dim]
        speech_info = output_lstm * speech_info.unsqueeze(1)  # [N, T, spk_emb_dim]

        out_spk_mask = self.fc_mask(speech_info)
        out_res_mask = torch.clamp(res_mask - out_spk_mask, 0, 1)

        spk_emb = self.fc_emb(speech_info)  # [N, T, spk_emb_dim]
        spk_emb = torch.mean(spk_emb, axis=1)  # [N, spk_emb_dim]
        spk_emb = self.fc_affine(spk_emb)  # [N, spk_emb_dim]

        weight_gate = self.fc_gate(speech_info)  # [N, T, spk_emb_dim]
        weight_gate = torch.mean(weight_gate, axis=1)  # [N, spk_emb_dim]

        out_emb = weight_gate * spk_emb + spk_info  # [N, spk_emb_dim]
        out_emb = out_emb / torch.norm(out_emb, p=2, dim=1, keepdim=True)

        out_res_mask = out_res_mask.permute(0, 2, 1).contiguous()  # [N, F, T]
        out_spk_mask = out_spk_mask.permute(0, 2, 1).contiguous()

        out = {"est_mask": out_spk_mask, "res_mask": out_res_mask, "spk_info": out_emb}

        return out


class RSAN(nn.Module):
    def __init__(
        self,
        n_channel=2,
        nfft=512,
        hop=128,
        n_layers=2,
        hidden_size=600,
        dropout=0,
        bidirectional=True,
        spk_emb_dim=128,
        block_size=30,
    ):
        super().__init__()

        self.ipd_extractor = IPDFeature(n_channel)

        self.module = RSAN_module(
            in_spat=(nfft // 2 + 1) * (n_channel - 1),
            n_bin=nfft // 2 + 1,
            n_layers=n_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            spk_emb_dim=spk_emb_dim,
        )

        self.torch_stft = partial(stft_torch, n_fft=nfft, hop_length=hop,)
        self.torch_istft = partial(istft_torch, n_fft=nfft, hop_length=hop,)

        self.spk_emb_dim = spk_emb_dim

        self.block_size = block_size

        self.threshold_res = 0.5

    def forward(self, wav):
        # wav: [N, C, L]

        N, C, L = wav.size()
        wav = wav.reshape(N * C, L)
        spec_complex = self.torch_stft(wav)
        mags, p = mag_phase(spec_complex)
        _, F, T = mags.size()
        mags = mags.reshape(N, C, F, T)
        p = p.reshape(N, C, F, T)

        spec = mags[:, 0, :, :]
        spat = self.ipd_extractor(p)  # [N, MF, T]

        # spec: [N, F, T]
        # spat: [N, MF, T]
        N, F, T = spec.shape

        n_block = T // self.block_size
        if T - n_block * self.block_size > self.block_size / 2:
            n_block = n_block + 1

        list_spk_info = []
        noise_info = torch.zeros([N, self.spk_emb_dim]).cuda()
        list_spk_info.append([noise_info])

        list_spk_mask = []

        res_mask_all = None

        for i_block in range(n_block):
            if i_block == n_block - 1:
                spec_block = spec[:, :, i_block * self.block_size :]
                spat_block = spat[:, :, i_block * self.block_size :]
            else:
                spec_block = spec[:, :, i_block * self.block_size : (i_block + 1) * self.block_size]
                spat_block = spat[:, :, i_block * self.block_size : (i_block + 1) * self.block_size]

            _, _, T_block = spec_block.shape
            res_mask = torch.ones([N, F, T_block]).cuda()

            result = self.module(spec_block, spat_block, res_mask, list_spk_info[0][-1])

            res_mask = result["res_mask"]
            noise_mask = result["est_mask"]
            noise_info = result["spk_info"]
            list_spk_info[0].append(noise_info)

            if i_block == 0:
                list_spk_mask.append(noise_mask)
            else:
                list_spk_mask[0] = torch.cat([list_spk_mask[0], noise_mask], axis=-1)

            value_res_mask = torch.mean(res_mask)

            i_spk = 1
            while value_res_mask > self.threshold_res:
                # add a new speaker
                if i_spk == len(list_spk_info):
                    list_spk_info.append([torch.zeros([N, self.spk_emb_dim]).cuda()])
                    list_spk_mask.append(torch.zeros((N, F, i_block * self.block_size)).cuda())

                result = self.module(spec_block, spat_block, res_mask, list_spk_info[i_spk][-1])

                res_mask = result["res_mask"]
                spk_mask = result["est_mask"]
                spk_info = result["spk_info"]

                list_spk_info[i_spk].append(spk_info)

                list_spk_mask[i_spk] = torch.cat([list_spk_mask[i_spk], spk_mask], axis=-1)

                i_spk += 1

            if res_mask_all is None:
                res_mask_all = res_mask
            else:
                res_mask_all = torch.cat([res_mask_all, res_mask], axis=-1)

        wav_spk = torch.zeros((N, len(list_spk_info), L)).cuda()
        spec_spk = torch.zeros((N, len(list_spk_info), F, T)).cuda()
        for i_mask, mask in enumerate(list_spk_mask):
            spec_spk[:, i_mask, ...] = mask * spec
            out_wav = self.torch_istft(
                [spec_spk[:, i_mask, ...], p[:, 0, :, :]], use_mag_phase=True
            )
            wav_spk[:, i_mask, ...] = out_wav

        result = {
            "wav_spk": wav_spk,
            "spec_spk": spec_spk,
            "res_mask": res_mask_all,
            "spk_emb": list_spk_info,
        }

        return result


def test_nnet():
    net = RSAN(n_channel=2, block_size=30)
    wav = torch.randn([3, 2, 16000 * 3])

    test_cuda = True

    if test_cuda:
        wav = wav.cuda()
        gpus = [0]
        est = nn.parallel.data_parallel(net.cuda(), (wav), gpus, gpus[0])
    else:
        est = net(wav)

    print('est["spec_spk"].shape: ', est["spec_spk"].shape)
    print('est["wav_spk"].shape: ', est["wav_spk"].shape)
    print('est["res_mask"].shape: ', est["res_mask"].shape)
    print('est["spk_emb"].shape: ', est["spk_emb"][0][0].shape)


if __name__ == "__main__":
    test_nnet()

