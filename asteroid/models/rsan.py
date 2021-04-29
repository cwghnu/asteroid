import json
import os
import torch
from torch import nn
from sklearn.cluster import KMeans

from asteroid import torch_utils
import asteroid_filterbanks as fb
from asteroid.engine.optimizers import make_optimizer
from asteroid_filterbanks.transforms import mag, apply_mag_mask
from asteroid.dsp.vad import ebased_vad
from asteroid.masknn.recurrent import SingleRNN
from asteroid.utils.torch_utils import pad_x_to_y


def make_model_and_optimizer(conf):
    """Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    enc, dec = fb.make_enc_dec("stft", **conf["filterbank"])
    masker = Chimera(enc.n_feats_out // 2, **conf["masknet"])
    model = Model(enc, masker, dec)
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    return model, optimizer

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
            n_bin*2+in_spat,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

        self.spk_emb_dim = spk_emb_dim

        self.fc_lstm = nn.Sequential(
            nn.Linear(hidden_size*2, spk_emb_dim),
            nn.Sigmoid()
        )

        self.fc_adpt = nn.Sequential(
            nn.Linear(spk_emb_dim, spk_emb_dim),
            nn.Sigmoid()
        )

        self.fc_mask = nn.Sequential(
            nn.Linear(spk_emb_dim, n_bin),
            nn.Sigmoid()
        )

        self.fc_gate = nn.Sequential(
            nn.Linear(spk_emb_dim, spk_emb_dim),
            nn.Sigmoid()
        )

        self.fc_emb = nn.Sequential(
            nn.Linear(spk_emb_dim, 50),
            nn.Linear(50, 50),
            nn.Linear(50, spk_emb_dim),
            nn.Sigmoid()
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

        output_lstm, _ = self.lstm(input_lstm) # [N, T, hidden_size*2]
        output_lstm = self.fc_lstm(output_lstm) # [N, T, spk_emb_dim]

        speech_info = self.fc_adpt(spk_info) # [spk_emb_dim]
        speech_info = output_lstm * speech_info.unsqueeze(1) # [N, T, spk_emb_dim]

        out_spk_mask = self.fc_mask(speech_info)
        out_res_mask = torch.clamp(res_mask - out_spk_mask, 0, 1)

        spk_emb = self.fc_emb(speech_info)  # [N, T, spk_emb_dim]
        spk_emb = torch.mean(spk_emb, axis=1)   # [N, spk_emb_dim]
        spk_emb = self.fc_affine(spk_emb)   # [N, spk_emb_dim]
        
        weight_gate = self.fc_gate(speech_info) # [N, T, spk_emb_dim]
        weight_gate = torch.mean(weight_gate, axis=1)   # [N, spk_emb_dim]
        
        out_emb = weight_gate * spk_emb + spk_info  # [N, spk_emb_dim]
        out_emb = out_emb / torch.norm(out_emb, p=2, dim=1, keepdim=True)

        out_res_mask = out_res_mask.permute(0, 2, 1).contiguous()   # [N, F, T]
        out_spk_mask = out_spk_mask.permute(0, 2, 1).contiguous()

        out = {
            "est_mask": out_spk_mask,
            "res_mask": out_res_mask,
            "spk_info": out_emb
        }

        return out

class RSAN(nn.Module):
    def __init__(
        self,
        in_spat=257,
        n_bin=257,
        n_layers=2,
        hidden_size=600,
        dropout=0,
        bidirectional=True,
        spk_emb_dim=128,
        block_size=80,
    ):
        super().__init__()

        self.module = RSAN_module(
            in_spat=in_spat,
            n_bin=n_bin,
            n_layers=n_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            spk_emb_dim=spk_emb_dim
        )

        self.spk_emb_dim = spk_emb_dim

        self.block_size = block_size

        self.threshold_res = 0.5

    def forward(self, spec, spat):
        # spec: [N, F, T]
        # spat: [N, F, T]
        N, F, T = spec.shape

        n_block = T//self.block_size
        if T - n_block*self.block_size > self.block_size/2:
            n_block = n_block + 1

        list_spk_info = []
        noise_info = torch.zeros([N, self.spk_emb_dim]).cuda()
        list_spk_info.append([noise_info])

        list_spk_mask = []

        res_mask_all = None

        for i_block in range(n_block):
            if i_block == n_block - 1:
                spec_block = spec[:, :, i_block*self.block_size:]
                spat_block = spat[:, :, i_block*self.block_size:]
            else:
                spec_block = spec[:, :, i_block*self.block_size:(i_block+1)*self.block_size]
                spat_block = spat[:, :, i_block*self.block_size:(i_block+1)*self.block_size]

            _, _, T_block = spec_block.shape
            res_mask = torch.ones([N, F, T_block]).cuda()
            
            result = self.module(spec_block, spat_block, res_mask, list_spk_info[0][-1])

            res_mask = result['res_mask']
            noise_mask = result['est_mask']
            noise_info = result['spk_info']
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
                    list_spk_mask.append(torch.zeros((N, F, i_block*self.block_size)).cuda())

                result = self.module(spec_block, spat_block, res_mask, list_spk_info[i_spk][-1])

                res_mask = result['res_mask']
                spk_mask = result['est_mask']
                spk_info = result['spk_info']

                list_spk_info[i_spk].append(spk_info)

                list_spk_mask[i_spk] = torch.cat([list_spk_mask[i_spk], spk_mask], axis=-1)

                i_spk += 1

            if res_mask_all is None:
                res_mask_all = res_mask
            else:
                res_mask_all = torch.cat([res_mask_all, res_mask], axis=-1)

        for i_mask, mask in enumerate(list_spk_mask):
            list_spk_mask[i_mask] = mask * spec

        result = {
            'spec_spk': list_spk_mask,
            'res_mask': res_mask_all,
            'spk_emb': list_spk_info,
        }

        return result


def test_nnet():
    net = RSAN(block_size=30)
    x_spec = torch.randn([3, 257, 300])
    x_spat = torch.randn([3, 257, 300])

    test_cuda = True

    if test_cuda:
        x_spec = x_spec.cuda()
        x_spat = x_spat.cuda()
        gpus = [0]
        est = nn.parallel.data_parallel(net.cuda(), (x_spec, x_spat), gpus, gpus[0])
    else:
        est = net(x_spec, x_spat)

    print('est["spec_spk"].shape: ', est["spec_spk"][0].shape)
    print('est["res_mask"].shape: ', est["res_mask"].shape)
    print('est["spk_emb"].shape: ', est["spk_emb"][0][0].shape)

if __name__ == "__main__":
    test_nnet()

