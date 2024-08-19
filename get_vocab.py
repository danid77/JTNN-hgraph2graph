import sys
import argparse 
from hgraph import *
from rdkit import Chem
from multiprocessing import Pool

def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        try:
            hmol = MolGraph(s)
        except Chem.rdchem.KekulizeException as e:
            print(f"Error kekulizing molecule: {s}, {e}")
            continue
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
        
    return vocab

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()

    data = [mol for line in sys.stdin for mol in line.split()[:2]] # 데이터를 받아옴 
    data = list(set(data)) # 중복제거 

    batch_size = len(data) // args.ncpu + 1 
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)] # 베치 시이즈 분할
 
    pool = Pool(args.ncpu) # 
    vocab_list = pool.map(process, batches) # 배치마다 process 함수 처리
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab] # 배치마다 처리한거 하나로 합침 
    vocab = list(set(vocab)) # 중복 제거

    for x,y in sorted(vocab):
        print(x, y)
        
        

# def process(data):
#     vocab = set()
#     for line in data:
#         s = line.strip("\r\n ")
#         try:
#             hmol = MolGraph(s)
#             for node, attr in hmol.mol_tree.nodes(data=True):
#                 smiles = attr['smiles']
#                 vocab.add(attr['label'])
#                 for i, s in attr['inter_label']:
#                     vocab.add((smiles, s))
#         except Chem.rdchem.KekulizeException as e:
#             print(f"Error kekulizing molecule: {s}, {e}")
#             continue
#     return vocab

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ncpu', type=int, default=1)
#     args = parser.parse_args()

#     data = [mol for line in sys.stdin for mol in line.split()[:2]]
#     data = list(set(data))

#     batch_size = len(data) // args.ncpu + 1
#     batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

#     pool = Pool(args.ncpu)
#     vocab_list = pool.map(process, batches)
#     pool.close()
#     pool.join()

#     vocab = set()
#     for v in vocab_list:
#         vocab.update(v)

#     with open('vocab.txt', 'w') as f:
#         for item in sorted(vocab):
#             if isinstance(item, tuple):
#                 f.write(f"{item[0]} {item[1]}\n")
#             else:
#                 f.write(f"{item}\n")