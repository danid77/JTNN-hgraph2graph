# pretraining generation running bash file

# 1. Extract substructure vocabulary from a given set of molecules:
python get_vocab.py --ncpu 16 < data/chembl/all.txt > vocab.txt

# 2. Preprocess training data:
python preprocess.py --train data/chembl/all.txt --vocab data/chembl/vocab.txt --ncpu 16 --mode single
mkdir train_processed
mv tensor* train_processed/

# 3. Train graph generation model
mkdir ckpt/chembl-pretrained
python train_generator.py --train train_processed/ --vocab data/chembl/vocab.txt --save_dir ckpt/chembl-pretrained

# 4. Sample molecules from a model checkpoint
python generate.py --vocab data/chembl/vocab.txt --model ckpt/chembl-pretrained/model.ckpt --nsamples 1000
