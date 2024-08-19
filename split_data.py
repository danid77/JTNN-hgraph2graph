import os
import pickle
from sklearn.model_selection import train_test_split


def load_tensors(file_path):
    with open(file_path, 'rb') as file:
        tensors = pickle.load(file)
    return tensors

def split_data(tensors, train_size=0.8, random_state=42):
    train_tensors, valid_tensors = train_test_split(tensors, train_size=train_size, random_state=random_state)
    return train_tensors, valid_tensors

def main():
    # 전체 텐서 파일 경로 로드
    tensor_files = [os.path.join('train_processed', f'tensors-{i}.pkl') for i in range(51)]

    # 모든 텐서 로드
    all_tensors = []
    for file in tensor_files:
        all_tensors.extend(load_tensors(file))

    # 데이터 분할
    train_tensors, valid_tensors = split_data(all_tensors)

    # 분할된 데이터를 파일로 저장
    os.makedirs('train_processed/data/train', exist_ok=True)
    os.makedirs('train_processed/data/valid', exist_ok=True)

    with open('train_processed/data/train/train_tensors.pkl', 'wb') as file:
        pickle.dump(train_tensors, file)

    with open('train_processed/data/valid/valid_tensors.pkl', 'wb') as file:
        pickle.dump(valid_tensors, file)

if __name__ == "__main__":
    main()