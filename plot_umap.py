from itertools import count
from pathlib import Path
import random
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
import glob
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import pylab
from umap import UMAP
# import plotly.express as px


speakers = ['BDL (M, L1)', 'CLB (F, L1)', 'ERMS (M, SP)', 'NJS (F, SP)', 'TXHC (M, CN)', 'LXC (F, CN)', 'YKWK (M, KR)', 'YDCK (F, KR)', 'ABA (M, AB)', 'ZHAA (F, AB)']
# baseline_speakers = ['FAC-B (TXHC + L1)', 'FAC-B (YKWK + L1)']
proposed_speakers = ['src_BDL_ref_TXHC (TXHC + L1(BDL))', 'src_CLB_ref_YKWK (YKWK + L1(CLB))']


all_speakers = speakers + proposed_speakers
markers = ["d" , "d", "o", "o", "^", "^", "p", "p", "P", "P"]
fac_markers = [ "s", "v"]


base_dir = '/mnt/data2/bhanu/datasets/dvec/GE2E_spkEmbed_step_5805000'
all_embedings = []
speaker_labels = []
for sp in all_speakers:
    speaker = sp.split(" ")[0]
    speaker_file_list = sorted(glob.glob(f"/mnt/data2/bhanu/datasets/dvec/GE2E_spkEmbed_step_5805000/{speaker}/*.npy"))
    print(f"Number of source utterances: {len(speaker_file_list)}.")
    t = np.load(speaker_file_list[0])
    cnt = 0
    for spf in speaker_file_list:
        if cnt > 18:
            break
        all_embedings.append(np.load(spf))
        speaker_labels.append(sp)
        cnt = cnt +1

print(np.array(all_embedings).shape)

print("Computing t-SNE embedding - speaker")
tsne_sp = UMAP(n_components=2, init='spectral', random_state=0)
speaker_tsne = tsne_sp.fit_transform(np.array(all_embedings))
print(speaker_tsne.shape)
colors =  mpl.cm.get_cmap('tab20')(np.arange(12))
plt.figure(figsize=(12,8))
speakers_all = all_speakers
markers_all = markers + fac_markers
for speaker, c, m in zip(speakers_all, colors, markers_all):
    X_speaker_embedding = speaker_tsne[np.where(speaker==np.array(speaker_labels))]
    print(X_speaker_embedding.shape, speaker)
    plt.scatter(X_speaker_embedding[:,0], X_speaker_embedding[:,1], label=speaker, marker=m, color=c)
    plt.text(X_speaker_embedding[-1,0], X_speaker_embedding[-1,1], speaker)

plt.legend()
plt.tight_layout()
plt.savefig("embed_viz/umap_SpeakerEmbeddings_ppg2ppg.png", format='png')

# print("Computing t-SNE embedding - speaker")
# tsne_sp = UMAP(n_components=3, init='random', random_state=0)
# speaker_tsne = tsne_sp.fit_transform(np.array(all_embedings))
# print(speaker_tsne.shape)
# colors =  mpl.cm.get_cmap('tab20')(np.arange(12))
# plt.figure(figsize=(12,8))
# fig = pylab.figure()
# ax = fig.add_subplot(111, projection = '3d')

# speakers_all = all_speakers
# markers_all = markers + fac_markers
# for speaker, c, m in zip(speakers_all, colors, markers_all):
#     X_speaker_embedding = speaker_tsne[np.where(speaker==np.array(speaker_labels))]
#     print(X_speaker_embedding.shape, speaker)
#     ax.scatter(X_speaker_embedding[:,0], X_speaker_embedding[:,1],X_speaker_embedding[:,2], label=speaker, marker=m, color=c)
#     ax.text(X_speaker_embedding[-1,0], X_speaker_embedding[-1,1],X_speaker_embedding[-1,2], speaker)

# # plt.legend()
# plt.tight_layout()
# plt.savefig("embed_viz/umap_3d_SpeakerEmbeddings_ppg2ppg.png", format='png')