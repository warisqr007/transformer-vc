import time
import sys
import os
import argparse
import torch
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
from conformer_ppg_model.build_ppg_model import load_ppg_model
from src.mel_decoder_mol_encAddlf0 import MelDecoderMOL
from src.mel_decoder_lsa import MelDecoderLSA
from src.rnn_ppg2mel import BiRnnPpg2MelModel
from src.transformer_bnftomel import Transformer
import pyworld
import librosa
import resampy
import soundfile as sf
from src.transformer_bnftomel import Transformer
from utils.f0_utils import get_cont_lf0
from utils.load_yaml import HpsYaml
import random
from vocoders.hifigan_model import load_hifigan_generator

from speaker_encoder.voice_encoder import SpeakerEncoder
from speaker_encoder.audio import preprocess_wav
from src import build_model
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import librosa.display
from skimage.transform import resize

def compute_spk_dvec(
    wav_path, weights_fpath="speaker_encoder/ckpt/pretrained_bak_5805000.pt",
):
    fpath = Path(wav_path)
    wav = preprocess_wav(fpath)
    # print('wac-shape',wav.shape)
    encoder = SpeakerEncoder(weights_fpath)
    spk_dvec = encoder.embed_utterance(wav)
    print(spk_dvec)
    return spk_dvec

def compute_spk_dvec1(
    mel, weights_fpath="speaker_encoder/ckpt/pretrained_bak_5805000.pt",
):
    wav = preprocess_wav(mel)
    encoder = SpeakerEncoder(weights_fpath)
    spk_dvec = encoder.embed_utterance(wav)
    print(spk_dvec)
    return spk_dvec

def compute_f0(wav, sr=16000, frame_period=10.0):
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(
        wav, sr, frame_period=frame_period, f0_floor=20.0, f0_ceil=600.0)
    return f0


def compute_mean_std(lf0):
    nonzero_indices = np.nonzero(lf0)
    mean = np.mean(lf0[nonzero_indices])
    std = np.std(lf0[nonzero_indices])
    return mean, std 


def f02lf0(f0):
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    return lf0


def get_converted_lf0uv(
    wav, 
    lf0_mean_trg, 
    lf0_std_trg,
    convert=True,
):
    f0_src = compute_f0(wav)
    if not convert:
        uv, cont_lf0 = get_cont_lf0(f0_src)
        lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
        return lf0_uv

    lf0_src = f02lf0(f0_src)
    lf0_mean_src, lf0_std_src = compute_mean_std(lf0_src)
    
    lf0_vc = lf0_src.copy()
    lf0_vc[lf0_src > 0.0] = (lf0_src[lf0_src > 0.0] - lf0_mean_src) / lf0_std_src * lf0_std_trg + lf0_mean_trg
    f0_vc = lf0_vc.copy()
    f0_vc[lf0_src > 0.0] = np.exp(lf0_vc[lf0_src > 0.0])
    
    uv, cont_lf0_vc = get_cont_lf0(f0_vc)
    lf0_uv = np.concatenate([cont_lf0_vc[:, np.newaxis], uv[:, np.newaxis]], axis=1)
    return lf0_uv


def build_ppg2mel_model(model_config, model_file, device):
    model_class = build_model(model_config["model_name"])
    ppg2mel_model = model_class(
        **model_config["model"]
    ).to(device)
    ckpt = torch.load(model_file, map_location=device)
    ppg2mel_model.load_state_dict(ckpt["model"])
    ppg2mel_model.eval()
    return ppg2mel_model

def build_transf_model(model_config, model_file, device):
    model_class = build_model(model_config["model_name"])
    ppg2mel_model = model_class(
        model_config["model"]
    ).to(device)
    ckpt = torch.load(model_file, map_location=device)
    ppg2mel_model.load_state_dict(ckpt["model"])
    ppg2mel_model.eval()
    return ppg2mel_model

@torch.no_grad()
def convert(args):
    device = 'cuda'
    ppg2mel_config = HpsYaml(args.ppg2mel_model_train_config) 
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    step = os.path.basename(args.ppg2mel_model_file)[:-4].split("_")[-1]

    # Build models
    print("Load PPG-model, PPG2Mel-model, Vocoder-model...")
    ppg_model = load_ppg_model(
        './conformer_ppg_model/en_conformer_ctc_att/config.yaml', 
        './conformer_ppg_model/en_conformer_ctc_att/24epoch.pth',
        device,
    )
    ppg2mel_model = build_transf_model(ppg2mel_config, args.ppg2mel_model_file, device) 
    hifigan_model = load_hifigan_generator(device)
    
    # Data related
    ref_wav_path = args.ref_wav_path
    ref_fid = os.path.basename(ref_wav_path)[:-4]
    # ref_spk_dvec = compute_spk_dvec(ref_wav_path)
    # ref_spk_dvec = torch.from_numpy(ref_spk_dvec).unsqueeze(0).to(device)
    # ref_wav, _ = librosa.load(ref_wav_path, sr=16000)
    # ref_lf0_mean, ref_lf0_std = compute_mean_std(f02lf0(compute_f0(ref_wav)))
    
    source_file_list = sorted(glob.glob(f"{args.src_wav_dir}/*.wav"))
    dest_file_list = sorted(glob.glob(f"{args.ref_wav_path}/*.wav"))
    print(f"Number of source utterances: {len(source_file_list)}.")
    # src_nos = np.arange(0,len(source_file_list))
    # temp = list(zip(source_file_list, src_nos))
    # random.Random(4).shuffle(temp)
    # source_file_list, src_nos = zip(*temp)
    # # res1 and res2 come out as tuples, and so must be converted to lists.
    # source_file_list, src_nos = list(source_file_list), list(src_nos)
    total_rtf = 0.0
    cnt = 0
    i=20
    # for src_wav_path in tqdm(source_file_list):
    for i in tqdm(range(21,41)):
        if i==40:
            break
        # Load the audio to a numpy array:
        # ref_spk_dvec = compute_spk_dvec(dest_file_list[src_nos[i]])
        src_wav_path = source_file_list[i]
        ref_spk_dvec = compute_spk_dvec(dest_file_list[i])
        ref_spk_dvec = torch.from_numpy(ref_spk_dvec).unsqueeze(0).to(device)
        src_wav, _ = librosa.load(src_wav_path, sr=16000)
        src_wav_tensor = torch.from_numpy(src_wav).unsqueeze(0).float().to(device)
        src_wav_lengths = torch.LongTensor([len(src_wav)]).to(device)
        ppg = ppg_model(src_wav_tensor, src_wav_lengths)

        # lf0_uv = get_converted_lf0uv(src_wav, ref_lf0_mean, ref_lf0_std, convert=True)
        # min_len = min(ppg.shape[1], len(lf0_uv))
        min_len = ppg.shape[1]

        ppg = ppg[:, :min_len]
        # lf0_uv = lf0_uv[:min_len]
        i = i+1
        
        start = time.time()
        if isinstance(ppg2mel_model, BiRnnPpg2MelModel):
            ppg_length = torch.LongTensor([ppg.shape[1]]).to(device)
            # logf0_uv=torch.from_numpy(lf0_uv).unsqueeze(0).float().to(device)
            mel_pred = ppg2mel_model(ppg, ppg_length, ref_spk_dvec)
        elif isinstance(ppg2mel_model, Transformer):
            mel_pred, att_ws = ppg2mel_model.inference(torch.squeeze(ppg), torch.squeeze(ref_spk_dvec))

        else:
            _, mel_pred, att_ws = ppg2mel_model.inference(
                ppg,
                spembs=ref_spk_dvec,
                use_stop_tokens=True,
            )

        # print(f'ppg : {ppg.shape} \n ppg_length : {ppg_length.shape} \n ref_spk_dvec : {ref_spk_dvec.shape} \n mel_pred : {mel_pred.shape} ')
        # if ppg2mel_config.data.min_max_norm_mel:
            # mel_min = ppg2mel_config.data.mel_min
            # mel_max = ppg2mel_config.data.mel_max
            # mel_pred = (mel_pred + 4.0) / 8.0 * (mel_max - mel_min) + mel_min
        
        mel_pred = mel_pred.unsqueeze(0)
        # print(mel_pred.shape,'--- mel shape',mel_pred)
        # test_spk_dvec(mel_pred[:,:,0:40])
        src_fid = os.path.basename(src_wav_path)[:-4]
        # wav_fname = f"{output_dir}/vc_{src_fid}_ref_{ref_fid}_step{step}.wav"
        # wav_fname = f"tsne/src_TXHC_ref_CLB/{src_fid}.wav"
        wav_fname = f"tsne/src_BDL_ref_YKWK/{src_fid}.wav"

        # print(wav_fname)
        # os.makedirs(wav_fname, exist_ok = True)
        mel_len = mel_pred.shape[0]
        rtf = (time.time() - start) / (0.01 * mel_len)
        total_rtf += rtf
        cnt += 1
        # continue
        y = hifigan_model(mel_pred.view(1, -1, 80).transpose(1, 2))

        # fig = plt.Figure()
        # canvas = FigureCanvas(fig)
        # ax = fig.add_subplot(111)
        # mel_pred = mel_pred.squeeze().cpu().numpy()
        # seconds_mel = mel_pred[:,::2]
        # print('melpred',mel_pred)
        # # seconds_mel = resize(mel_pred,(mel_pred.shape[0],mel_pred.shape[1]/2),order=1)
        # print('mel-pred-shape',mel_pred.shape)
        # print('mel-pred-shape',seconds_mel.shape)
        # p = librosa.display.specshow(librosa.amplitude_to_db(mel_pred, ref=np.max), ax=ax, y_axis='mel', x_axis='time')
        # print(p)
        # fig.savefig(f"{output_dir}/spec.png")
        # p = librosa.display.specshow(librosa.amplitude_to_db(seconds_mel, ref=np.max), ax=ax, y_axis='mel', x_axis='time')
        # fig.savefig(f"{output_dir}/spec_1.png")
        sf.write(wav_fname, y.squeeze().cpu().numpy(), 24000, "PCM_16")
        # compute_spk_dvec1(y.squeeze().cpu().numpy())
    print("RTF:")
    print(total_rtf / cnt)


def get_parser():
    parser = argparse.ArgumentParser(description="Conversion from wave input")
    parser.add_argument(
        "--src_wav_dir",
        type=str,
        default=None,
        required=True,
        help="Source wave directory.",
    )
    parser.add_argument(
        "--ref_wav_path",
        type=str,
        required=True,
        help="Reference wave file path.",
    )
    parser.add_argument(
        "--ppg2mel_model_train_config", "-c",
        type=str,
        default=None,
        required=True,
        help="Training config file (yaml file)",
    )
    parser.add_argument(
        "--ppg2mel_model_file", "-m",
        type=str,
        default=None,
        required=True,
        help="ppg2mel model checkpoint file path"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="vc_gens_vctk_oneshot",
        help="Output folder to save the converted wave."
    )
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
