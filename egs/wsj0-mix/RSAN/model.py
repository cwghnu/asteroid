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
        in_chan,
        n_bin=257,
        n_layers=2,
        hidden_size=600,
        dropout=0,
        bidirectional=True,
        spk_emb_dim=128,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            in_chan,
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

        spec = spec.permute(0, 2, 1)
        spat = spat.permute(0, 2, 1)
        res_mask = res_mask.permute(0, 2, 1)

        input_lstm = torch.cat([spec, spat, res_mask], axis=2)  # [N, T, F*3]

        output_lstm = self.lstm(input_lstm) # [N, T, hidden_size*2]
        output_lstm = self.fc_lstm(output_lstm) # [N, T, spk_emb_dim]

        speech_info = self.fc_adpt(spk_info) # [spk_emb_dim]
        speech_info = output_lstm * speech_info # [N, T, spk_emb_dim]

        out_spk_mask = self.fc_mask(speech_info)
        out_res_mask = torch.clamp(res_mask - out_spk_mask, 0, 1)

        spk_emb = self.fc_emb(speech_info)  # [N, T, spk_emb_dim]
        spk_emb = torch.mean(spk_emb)   # [N, spk_emb_dim]
        spk_emb = self.fc_affine(spk_emb)   # [N, spk_emb_dim]
        
        weight_gate = self.fc_gate(speech_info) # [N, T, spk_emb_dim]
        weight_gate = torch.mean(weight_gate, axis=1)   # [N, spk_emb_dim]
        
        out_emb = weight_gate * spk_emb + spk_info  # [N, spk_emb_dim]
        out_emb = torch.norm(out_emb, p=2, dim=1)

        out = {
            "est_mask": out_spk_mask,
            "res_mask": out_res_mask,
            "spk_info": out_emb
        }

        return out

class RSAN(nn.Module):
    def __init__(
        self,
        in_chan,
        n_bin=257,
        n_layers=2,
        hidden_size=600,
        dropout=0,
        bidirectional=True,
        spk_emb_dim=128,
    ):

        self.module = RSAN_module(
            in_chan,
            n_bin=n_bin,
            n_layers=n_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            spk_emb_dim=spk_emb_dim
        )

    def forward(self, spec, spat):


class Chimera(nn.Module):
    def __init__(
        self,
        in_chan,
        n_src,
        rnn_type="lstm",
        n_layers=2,
        hidden_size=600,
        bidirectional=True,
        dropout=0.3,
        embedding_dim=20,
        take_log=False,
        EPS=1e-8,
    ):
        super().__init__()
        self.input_dim = in_chan
        self.n_src = n_src
        self.take_log = take_log
        # RNN common
        self.embedding_dim = embedding_dim
        self.rnn = SingleRNN(
            rnn_type,
            in_chan,
            hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        rnn_out_dim = hidden_size * 2 if bidirectional else hidden_size
        # Mask heads
        self.mask_layer = nn.Linear(rnn_out_dim, in_chan * self.n_src)
        self.mask_act = nn.Sigmoid()  # sigmoid or relu or softmax
        # DC head
        self.embedding_layer = nn.Linear(rnn_out_dim, in_chan * embedding_dim)
        self.embedding_act = nn.Tanh()  # sigmoid or tanh
        self.EPS = EPS

    def forward(self, input_data):
        batch, _, n_frames = input_data.shape
        if self.take_log:
            input_data = torch.log(input_data + self.EPS)
        # Common net
        out = self.rnn(input_data.permute(0, 2, 1))
        out = self.dropout(out)

        # DC head
        proj = self.embedding_layer(out)  # batch, time, freq * emb
        proj = self.embedding_act(proj)
        proj = proj.view(batch, n_frames, -1, self.embedding_dim).transpose(1, 2)
        # (batch, freq * frames, emb)
        proj = proj.reshape(batch, -1, self.embedding_dim)
        proj_norm = torch.norm(proj, p=2, dim=-1, keepdim=True)
        projection_final = proj / (proj_norm + self.EPS)

        # Mask head
        mask_out = self.mask_layer(out).view(batch, n_frames, self.n_src, self.input_dim)
        mask_out = mask_out.permute(0, 2, 3, 1)
        mask_out = self.mask_act(mask_out)
        return projection_final, mask_out


class Model(nn.Module):
    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        final_proj, mask_out = self.masker(mag(tf_rep))
        return final_proj, mask_out

    def separate(self, x):
        """ Separate with mask-inference head, output waveforms """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        proj, mask_out = self.masker(mag(tf_rep))
        masked = apply_mag_mask(tf_rep.unsqueeze(1), mask_out)
        wavs = torch_utils.pad_x_to_y(self.decoder(masked), x)
        dic_out = dict(tfrep=tf_rep, mask=mask_out, masked_tfrep=masked, proj=proj)
        return wavs, dic_out

    def dc_head_separate(self, x):
        """ Cluster embeddings to produce binary masks, output waveforms """
        kmeans = KMeans(n_clusters=self.masker.n_src)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        mag_spec = mag(tf_rep)
        proj, mask_out = self.masker(mag_spec)
        active_bins = ebased_vad(mag_spec)
        active_proj = proj[active_bins.view(1, -1)]
        #
        bin_clusters = kmeans.fit_predict(active_proj.cpu().data.numpy())
        # Create binary masks
        est_mask_list = []
        for i in range(self.masker.n_src):
            # Add ones in all inactive bins in each mask.
            mask = ~active_bins
            mask[active_bins] = torch.from_numpy((bin_clusters == i)).to(mask.device)
            est_mask_list.append(mask.float())  # Need float, not bool
        # Go back to time domain
        est_masks = torch.stack(est_mask_list, dim=1)
        masked = apply_mag_mask(tf_rep, est_masks)
        wavs = pad_x_to_y(self.decoder(masked), x)
        dic_out = dict(tfrep=tf_rep, mask=mask_out, masked_tfrep=masked, proj=proj)
        return wavs, dic_out


def load_best_model(train_conf, exp_dir):
    """Load best model after training.

    Args:
        train_conf (dict): dictionary as expected by `make_model_and_optimizer`
        exp_dir(str): Experiment directory. Expects to find
            `'best_k_models.json'` of `checkpoints` directory in it.

    Returns:
        nn.Module the best (or last) pretrained model according to the val_loss.
    """
    # Create the model from recipe-local function
    model, _ = make_model_and_optimizer(train_conf)
    try:
        # Last best model summary
        with open(os.path.join(exp_dir, "best_k_models.json"), "r") as f:
            best_k = json.load(f)
        best_model_path = min(best_k, key=best_k.get)
    except FileNotFoundError:
        # Get last checkpoint
        all_ckpt = os.listdir(os.path.join(exp_dir, "checkpoints/"))
        all_ckpt = [
            (ckpt, int("".join(filter(str.isdigit, os.path.basename(ckpt)))))
            for ckpt in all_ckpt
            if ckpt.find("ckpt") >= 0
        ]
        all_ckpt.sort(key=lambda x: x[1])
        best_model_path = os.path.join(exp_dir, "checkpoints", all_ckpt[-1][0])
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location="cpu")
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint["state_dict"], model)
    model.eval()
    return model
