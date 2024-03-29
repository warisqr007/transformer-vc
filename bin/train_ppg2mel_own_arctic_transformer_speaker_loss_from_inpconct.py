import torch
from torch.utils.data import DataLoader
import numpy as np
from src.solver import BaseSolver
# from src.data_load import VcDataset, VcCollate
from src.data_load import OneshotVcDataset, MultiSpkVcCollate, OneshotArciticVcDataset
from src.transformer_bnftomel import Transformer
from src.optim import Optimizer
import torch_optimizer as optim
from src.util import human_format, feat_to_fig
from src.loss_fn import MaskedMSELoss
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import (
    Tacotron2Loss as TransformerLoss,  # noqa: H301
)
from vocoders.hifigan_model import load_hifigan_generator
from speaker_encoder.audio import preprocess_wav
from speaker_encoder.voice_encoder import SpeakerEncoder
from joblib import Parallel, delayed

class Solver(BaseSolver):
    """Customized Solver."""
    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.best_loss = np.inf
        self.optimizer_dict = ''
        self.tmp_step_val = 0

    def fetch_data(self, data):
        """Move data to device"""
        data = [i.to(self.device) for i in data]
        return data

    def load_data(self):
        """ Load data for training/validation/plotting."""
        train_dataset = OneshotArciticVcDataset(
            meta_file=self.config.data.train_fid_list,
            arctic_ppg_dir = self.config.data.arctic_ppg_dir,
            arctic_f0_dir = self.config.data.arctic_f0_dir,
            arctic_wav_dir = self.config.data.arctic_wav_dir,
            arctic_spk_dvec_dir = self.config.data.arctic_spk_dvec_dir,
            ppg_file_ext=self.config.data.ppg_file_ext,
            min_max_norm_mel=self.config.data.min_max_norm_mel,
            mel_min=self.config.data.mel_min,
            mel_max=self.config.data.mel_max,
        )
        dev_dataset = OneshotArciticVcDataset(
            meta_file=self.config.data.dev_fid_list,
            arctic_ppg_dir = self.config.data.arctic_ppg_dir,
            arctic_f0_dir = self.config.data.arctic_f0_dir,
            arctic_wav_dir = self.config.data.arctic_wav_dir,
            arctic_spk_dvec_dir = self.config.data.arctic_spk_dvec_dir,
            ppg_file_ext=self.config.data.ppg_file_ext,
            min_max_norm_mel=self.config.data.min_max_norm_mel,
            mel_min=self.config.data.mel_min,
            mel_max=self.config.data.mel_max,
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            num_workers=self.paras.njobs,
            shuffle=True,
            batch_size=self.config.hparas.batch_size,
            pin_memory=False,
            drop_last=True,
            collate_fn=MultiSpkVcCollate(n_frames_per_step=1,
                                         f02ppg_length_ratio=1,
                                         use_spk_dvec=True),
        )
        self.dev_dataloader = DataLoader(
            dev_dataset,
            num_workers=self.paras.njobs,
            shuffle=False,
            batch_size=self.config.hparas.batch_size,
            pin_memory=False,
            drop_last=False,
            collate_fn=MultiSpkVcCollate(n_frames_per_step=1,
                                         f02ppg_length_ratio=1,
                                         use_spk_dvec=True),
        )
        msg = "Have prepared training set and dev set."
        self.verbose(msg)
    
    def load_pretrained_params(self):
        prefix = "ppg2mel_model"
        ignore_layers = ["ppg2mel_model.spk_embedding.weight"]
        pretrain_model_file = self.config.data.pretrain_model_file
        pretrain_ckpt = torch.load(
            pretrain_model_file, map_location=self.device
        )
        model_dict = self.model.state_dict()
        # print(model_dict.keys())
        # # 1. filter out unnecessrary keys
        # print(pretrain_ckpt['model'].keys())
        # # print(pretrain_ckpt['optimizer'])
        # print(pretrain_ckpt['global_step'])
        # print(pretrain_ckpt['loss'])
        
        
        # print('decoder.decoders.4.src_attn.linear_q.weight'.split(".", maxsplit=1))
        # for k,v in pretrain_ckpt.items():
        #     print(k['model'])
        #     print(k['model'][0].split(".", maxsplit=1))
        # pretrain_dict = {k.split(".", maxsplit=1)[1]: v 
        #                  for k, v in pretrain_ckpt.items() if "spk_embedding" not in k 
        #                     and "wav2ppg_model" not in k and "reduce_proj" not in k}
        # ----------
        pretrain_dict = pretrain_ckpt['model']
        self.best_loss = pretrain_ckpt['loss']
        self.step = pretrain_ckpt['global_step']
        self.tmp_step_val = self.step
        self.optimizer_dict = pretrain_ckpt['optimizer']
        assert len(pretrain_dict.keys()) == len(model_dict.keys())

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrain_dict)

        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

    def set_model(self):
        """Setup model and optimizer"""
        # Model
        self.model = Transformer(self.config["model"]).to(self.device)
        # self.model = Transformer(self.config["model"])
        # self.model= torch.nn.DataParallel(self.model)
        # self.model.to(self.device)
        if "pretrain_model_file" in self.config.data:
            self.load_pretrained_params()

        # model_params = [{'params': self.model.spk_embedding.weight}]
        model_params = [{'params': self.model.parameters()}]
        args = self.config["model"]
        # Loss criterion
        # self.loss_criterion = MaskedMSELoss()
        # Optimizer
        self.optimizer = Optimizer(model_params, **self.config["hparas"])
        if "pretrain_model_file" in self.config.data:
            self.optimizer.load_opt_state_dict(self.optimizer_dict)
        # self.optimizer = optim.Lamb(model_params, lr=0.001,clamp_value=1)
        self.verbose(self.optimizer.create_msg())

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()
    
    # @torch.no_grad()
    # def compute_secondary_models(self,mel_pred):
    #     device = 'cuda'
    #     hifigan_model = load_hifigan_generator(device)
    #     y = hifigan_model(mel_pred.view(self.config.hparas.batch_size, -1, 80).transpose(1, 2))
    #     y = y.cpu().numpy()
    #     y = np.squeeze(y,axis=1)
    #     weights_fpath="speaker_encoder/ckpt/pretrained_bak_5805000.pt"
    #     spkdv = []
    #     # print(y.shape)
    #     for i in range(y.shape[0]):
    #         fpath = y[i,:]
    #         wav = preprocess_wav(fpath)
    #         encoder = SpeakerEncoder(weights_fpath)
    #         spk_dvec = encoder.embed_utterance(wav)
    #         spkdv.append(spk_dvec)
    #     spkdvs = torch.from_numpy(np.array(spkdv)).to(device)
    #     # print(spkdvs.shape,type(spkdvs))
    #     return spkdvs
    # @torch.no_grad()
    # def compute_secondary_models(self,mel_pred):
    #     hifigan_model = load_hifigan_generator('cuda')
    #     weights_fpath="speaker_encoder/ckpt/pretrained_bak_5805000.pt"
    #     encoder = SpeakerEncoder(weights_fpath)
    #     spkdv = []
    #     # print(mel_pred.shape)
    #     for j in range(mel_pred.shape[0]):
    #         mel_s = mel_pred[j,:,:]
    #         y = self.hifigan_model(mel_s.view(1, -1, 80).transpose(1, 2))
    #         y = y.cpu().numpy()
    #         y = np.squeeze(y,axis=1)
    #         # print(y.shape)
    #         fpath = y[0,:]
    #         # print(fpath.shape)
    #         wav = preprocess_wav(fpath)
    #         spk_dvec = self.encoder.embed_utterance(wav)
    #         spkdv.append(spk_dvec)
    #     spkdvs = torch.from_numpy(np.array(spkdv)).to('cuda')
    #     torch.cuda.empty_cache()
    #     # print(spkdvs.shape,type(spkdvs))
    #     return spkdvs
    
    # @torch.no_grad()
    # def funcc(self,mel_s,encoder,lwav):
    #     print('In train',mel_s.shape)
    #     spk_dvec = encoder.embed_mel(mel_s,lwav)
    #     return(spk_dvec)
    # @torch.no_grad()
    # def compute_secondary_models(self,mel_pred,lwav):
    #     # spkdv = []
    #     # print(mel_pred.shape)
    #     weights_fpath="speaker_encoder/ckpt/pretrained_bak_5805000.pt"
    #     encoder = SpeakerEncoder(weights_fpath)
    #     mel_pred = mel_pred.cpu().detach().numpy()
    #     data = Parallel(n_jobs=2,backend="threading")(delayed(self.funcc)(mel_s,encoder,lwav) for mel_s in mel_pred)
    #     spkdvs = torch.from_numpy(np.array(data)).to('cuda')
    #     torch.cuda.empty_cache()
    #     print(spkdvs.shape,type(spkdvs))
    #     return spkdvs
    @torch.no_grad()
    def funcc(self,mel_s,hifigan_model,encoder):
        y = hifigan_model(mel_s.view(1, -1, 80).transpose(1, 2))
        y = y.cpu().numpy()
        y = np.squeeze(y,axis=1)
        # print(y.shape)
        fpath = y[0,:]
        # print(fpath.shape)
        wav = preprocess_wav(fpath)
        spk_dvec = encoder.embed_utterance(wav)
        return(spk_dvec)
    @torch.no_grad()
    def compute_secondary_models(self,mel_pred):
        # spkdv = []
        # print(mel_pred.shape)
        hifigan_model = load_hifigan_generator('cuda')
        weights_fpath="speaker_encoder/ckpt/pretrained_bak_5805000.pt"
        encoder = SpeakerEncoder(weights_fpath)
        data = Parallel(n_jobs=16,backend="threading")(delayed(self.funcc)(mel_s,hifigan_model,encoder) for mel_s in mel_pred)
        spkdvs = torch.from_numpy(np.array(data)).to('cuda')
        torch.cuda.empty_cache()
        # print(spkdvs.shape,type(spkdvs))
        return spkdvs
    def exec(self):
        self.verbose("Total training steps {}.".format(
            human_format(self.max_step)))

        mel_loss = None
        n_epochs = 0
        mseloss = torch.nn.MSELoss()
        # Set as current time
        self.timer.set()
        
        while self.step < self.max_step:
            for data in self.train_dataloader:
                # Pre-step: updata lr_rate and do zero_grad
                # self.optimizer.zero_grad()
                lr_rate = self.optimizer.pre_step(self.step)
                total_loss = 0
                # data to device
                ppgs, lf0_uvs, mels, in_lengths, \
                    out_lengths, spk_ids, _, lwav = self.fetch_data(data)
                self.timer.cnt("rd")
                loss, after_outs, before_outs, ys, olens = self.model(
                    xs=ppgs,
                    ilens= in_lengths,
                    ys=mels,
                    olens=out_lengths,
                    logf0_uv=lf0_uvs,
                    spembs=spk_ids,
                )
                # loss = self.loss_criterion(mel_pred, mels, out_lengths)
                # loss, after_outs, before_outs, logits, ys, labels, olens
                # loss = l1_loss + l2_loss

                # spkdvs = self.compute_secondary_models(before_outs)
                # spkd_loss = mseloss(spkdvs,spk_ids)
                # mel_loss = loss.cpu().item()
                # spk_loss = spkd_loss.cpu().item()
                # # print('mel-loss',mel_loss)
                # # print('spk-loss',spk_loss)
                # # print(lr_rate)
                # loss = loss + 50*spkd_loss
                self.timer.cnt("fw")

                # Back-prop
                grad_norm = self.backward(loss)
                self.step += 1
                # Logger
                if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
                    # self.progress("Tr stat | Loss - {:.4f} | Mel - {:.4f} | Spk-loss - {:.4f} | Grad. Norm - {:.2f} | {}"
                    #               .format(loss.cpu().item(),mel_loss,spk_loss, grad_norm, self.timer.show()))
                    self.progress("Tr stat | Loss - {:.4f} | Grad. Norm - {:.2f} | {}"
                                  .format(loss.cpu().item(), grad_norm, self.timer.show()))
                    self.write_log('loss', {'tr': loss})

                # Validation
                if (self.step == 0) or (self.step % self.valid_step == 0) or (self.step==self.tmp_step_val + 1):
                    self.validate()

                # End of step
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()
                self.timer.set()
                if self.step > self.max_step:
                    break
            n_epochs += 1
        self.log.close()
    

    
    def validate(self):
        self.model.eval()
        dev_loss = 0.0
        mseloss = torch.nn.MSELoss()
        for i, data in enumerate(self.dev_dataloader):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.dev_dataloader)))
            # Fetch data
            # ppgs, lf0_uvs, mels, lengths = self.fetch_data(data)
            ppgs, lf0_uvs, mels, in_lengths, \
                out_lengths, spk_ids, _, _ = self.fetch_data(data)

            with torch.no_grad():
                loss, after_outs, before_outs, ys, olens = self.model(
                    xs=ppgs,
                    ilens= in_lengths,
                    ys=mels,
                    olens=out_lengths,
                    logf0_uv=lf0_uvs,
                    spembs=spk_ids,
                )
                # spkdvs = self.compute_secondary_models(before_outs)
                # spkd_loss = mseloss(spkdvs,spk_ids)
                # loss = loss + spkd_loss

                # loss = self.loss_criterion(mel_pred, mels, out_lengths)
                # loss, after_outs, before_outs, logits, ys, labels, olens
                # loss = l1_loss + l2_loss
                # loss = self.loss_criterion(mel_pred, mels, out_lengths)
                dev_loss += loss.cpu().item()

        dev_loss = dev_loss / (i + 1)
        self.save_checkpoint(f'step_{self.step}.pth', 'loss', dev_loss, show_msg=False)
        if dev_loss < self.best_loss:
            self.best_loss = dev_loss
            self.save_checkpoint(f'best_loss_step_{self.step}.pth', 'loss', dev_loss)
        self.write_log('loss', {'dv_loss': dev_loss})

        # Resume training
        self.model.train()

