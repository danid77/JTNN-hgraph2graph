from random import random
from optparse import OptionParser
import csv  # CSV 모듈 임포트

import rdkit
import torch
import time
import sys
sys.path.append('/home/lnp01/Desktop/icml18-jtnn')
from jtnn import *
from rdkit import Chem

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-n", "--nsample", dest="nsample")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts, args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
nsample = int(opts.nsample)
stereo = True if int(opts.stereo) == 1 else False

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
load_dict = torch.load(opts.model_path)
missing = {k: v for k, v in list(model.state_dict().items()) if k not in load_dict}
load_dict.update(missing)
model.load_state_dict(load_dict)
model = model.cuda()

query_list = {
    "Chem001": "CC1(C)CC(C)(C)CC(C1)N(C)C(=O)c(c2)ncc(c23)[nH]cnc3=O",
}

# CSV 파일 초기 설정
with open('generated_molecules01.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Query_ID", "Sampled_SMILES", "Sample_ID"])  # 헤더 쓰기

time_log = open('timelog01.txt', 'w')

for q_id, q_smi in query_list.items():
    start_time = time.time()
    sampled_list = []
    while True:
        try:        
            smi_tmp = model.reconstruct_scaled(
                q_smi,
                scale=random(),
                prob_decode=False,
                debug=False
            )
        except IndexError:       
            continue 

        if not smi_tmp:
            continue

        smi_tmp = Chem.MolToSmiles(Chem.MolFromSmiles(smi_tmp))

        if not smi_tmp in sampled_list:
            sampled_list.append(smi_tmp)
            print('succeed >>', smi_tmp, len(sampled_list))
        
        if len(sampled_list) == nsample:
            break
            
    end_time = time.time()
    execution_time = end_time - start_time

    hours = int(execution_time / 3600)
    minutes = int((execution_time % 3600) / 60)
    seconds = int((execution_time % 3600) % 60)
    
    time_log.write(f"{q_id}수행 시간: {hours}시간 {minutes}분 {seconds}초\n")

    with open(f"{q_id}_result.smi", 'w') as wf:
        wf.write(f"{q_smi} {q_id}\n")
        for idx, s_li in enumerate(sampled_list):
            wf.write('%s %s\n' % (s_li, f"{q_id}-S{str(idx + 1).zfill(3)}"))

    # CSV 파일에 결과 추가
    with open('generated_molecules01.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for idx, s_li in enumerate(sampled_list):
            sample_id = f"{q_id}-S{str(idx + 1).zfill(3)}"
            csv_writer.writerow([q_id, s_li, sample_id])  # 생성된 분자를 CSV에 추가

