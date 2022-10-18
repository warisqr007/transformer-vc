import glob2
import random
import numpy as np
from sklearn.model_selection import train_test_split

wav_file_list = glob2.glob(f"/mnt/data1/waris/datasets/data/arctic_dataset/all_data_for_ac_vc_train/**/*.wav")

ids = []
for t in wav_file_list:
    spkr = t.split('.')[0].split('/')[-3]
    fid = t.split('.')[0].split('/')[-1]
    wav  = t.split('.')[0].split('/')[-2]
    # with open('/path/to/filename.txt', mode='wt', encoding='utf-8') as myfile:

    ids.append(f'{spkr}/{fid}')

ids = np.array(ids)
np.random.shuffle(ids)

data_train, data_test, labels_train, labels_test = train_test_split(ids, ids, test_size=0.05, random_state=42)

with open('train.txt', mode='wt', encoding='utf-8') as myfile:
    for s in data_train:
        myfile.write(s)
        myfile.write('\n')
with open('dev.txt', mode='wt', encoding='utf-8') as myfile:
    for s in data_test:
        myfile.write(s)
        myfile.write('\n')