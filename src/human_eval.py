from src import config
from src.dataset import IfRaisesDataset
from src.preprocess import load_tokenizer

import numpy as np

if __name__ == '__main__':


    dataset_path = 'shared_resources/dataset_preprocessed_1000.pkl'
    # dataset_path = config.dataset_preprocessed_path
    tokenizer = load_tokenizer(config.model_input_len)

    dataset = IfRaisesDataset(dataset_path, tokenizer=tokenizer, fraction=0.1)
    for i, sample in enumerate(dataset):
        pass

    human = np.array(dataset.human_preds)
    real = np.array(dataset.human_gt)

    print(f'correct: {np.sum(human == real)} / {len(human)} ({np.sum(human == real) / len(human)}) ')

    np.unique(human - real, return_counts=True)
