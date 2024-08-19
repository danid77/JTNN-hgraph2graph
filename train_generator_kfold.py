import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset # Dataset 추가
from torch.cuda.amp import GradScaler, autocast # 추가
from sklearn.model_selection import KFold
import random
import pickle # 추가

import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
import wandb
import networkx as nx

from hgraph import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("PyTorch CUDA Version:", torch.version.cuda)
# print("CUDA Available:", torch.cuda.is_available())
# print("CUDA Device Count:", torch.cuda.device_count())
# print("Current CUDA Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# 환경 변수 설정
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 사용할 GPU 번호로 설정

# CUDA 메모리 초기화
torch.cuda.empty_cache()

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=256) # 250
parser.add_argument('--embed_size', type=int, default=256) # 250
parser.add_argument('--batch_size', type=int, default=50)
# parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0) # 0

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--print_iter', type=int, default=50) # 배치사이즈와 동일하게 
parser.add_argument('--save_iter', type=int, default=100000)

args = parser.parse_args()
print(args)

# wandb
wandb.login()
wandb.init(project="jtnn_view", entity="seungbeom_jin")

torch.manual_seed(args.seed)
random.seed(args.seed)

# vocab.txt 로드 및 처리
vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

# 모델 정의
# model = HierVAE(args).cuda()
model = HierVAE(args).to(device)
# model = nn.DataParallel(model, device_ids=[0, 1, 2])  # DataParallel로 모델을 감쌈
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scaler = GradScaler()
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

if args.load_model:
    print('continuing from checkpoint ' + args.load_model)
    model_state, optimizer_state, total_step, beta = torch.load(args.load_model)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
else:
    total_step = beta = 0

# 파라미터 및 그래디언트 노름 계산 함수
param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

# meters = torch.zeros(6).to(device)
total_step = 0
beta = 0.0

# WandB 설정
wandb.config.update(args)
wandb.watch(model, log="all")

# 각 .pkl 파일을 하나의 샘플로 간주하여 로드
pkl_files = [os.path.join(args.train, f"tensors-{i}.pkl") for i in range(56)]
data = []
for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        data.append(pickle.load(f))

# k-fold cross validation을 위한 데이터셋 분할
kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
print('')
print('training start!')
print('')
print('-------------------------------------------------------------------------------------------------------------------------------------------------------')

for fold, (train_index, val_index) in enumerate(kf.split(data)):
    print(f'Fold {fold+1}/{args.n_splits}')
    
    train_data = [data[i] for i in train_index]
    val_data = [data[i] for i in val_index]
    
    # DataLoader 설정
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)  
    
    for epoch in range(args.epoch):
        model.train()
        meters = torch.zeros(6).to(device)
    # beta = 0.0

        # for batch in tqdm(dataset):
        for batch in tqdm(train_loader):
            total_step += 1
            model.zero_grad()
            
            
            # PyTorch에서 Automatic Mixed Precision (AMP) 기능을 사용하는 방법입니다. 
            # AMP는 모델 학습 시 16비트와 32비트 부동 소수점 연산을 혼합하여 사용함으로써 GPU 메모리 사용량을 줄이고 
            # 연산 속도를 높일 수 있습니다.
            with autocast(enabled=True):
                loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)

            # 원래 코드
            # loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            # optimizer.step()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            meters += torch.tensor([kl_div, loss, wacc * 100, iacc * 100, tacc * 100, sacc * 100]).to(device)

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % 
                    (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))
                sys.stdout.flush()
                
                # WandB 로그 기록
                wandb.log({
                    "batch": total_step,
                    "epoch" : epoch,
                    "fold": fold + 1,
                    "beta": beta,
                    "KL Divergence": meters[0],
                    "Loss": meters[1],
                    "Word Accuracy": meters[2],
                    "Instance Accuracy": meters[3],
                    "Topology Accuracy": meters[4],
                    "Assembly Accuracy": meters[5],
                    "Parameter Norm": param_norm(model),
                    "Gradient Norm": grad_norm(model)
                })
                
                meters *= 0
            
            if total_step % args.save_iter == 0:
                ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
                torch.save(ckpt, os.path.join(args.save_dir, f"model_fold{fold+1}_step{total_step}.ckpt"))

            if total_step % args.anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
                beta = min(args.max_beta, beta + args.step_beta)
                # Validation 단계 추가 (선택 사항)
                
        model.eval()
        val_meters = torch.zeros(6).to(device)
        with torch.no_grad():
            for batch in tqdm(val_loader):
                loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)
                val_meters += torch.tensor([kl_div, loss, wacc * 100, iacc * 100, tacc * 100, sacc * 100]).to(device)
        
        val_meters /= len(val_loader)
        print(f"Fold {fold+1} Validation - KL: {val_meters[0]}, Loss: {val_meters[1]}, Word: {val_meters[2]}, Instance: {val_meters[3]}, Topo: {val_meters[4]}, Assm: {val_meters[5]}")
        
        # WandB validation 로그 기록
        wandb.log({
            "fold": fold + 1,
            "Validation KL Divergence": val_meters[0],
            "Validation Loss": val_meters[1],
            "Validation Word Accuracy": val_meters[2],
            "Validation Instance Accuracy": val_meters[3],
            "Validation Topology Accuracy": val_meters[4],
            "Validation Assembly Accuracy": val_meters[5]
        })
