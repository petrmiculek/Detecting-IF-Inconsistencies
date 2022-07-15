# Global config module

# Tokenization
tokens_length_unit = 32  # 256
model_input_len = 3 * tokens_length_unit  # cond, raise, pre-context

""" Paths """
# Word Embedding
fasttext_path = 'shared_resources/pretrained_fasttext/embed_if_32.mdl'

""" Dataset """
dataset_preprocessed_path = 'shared_resources/dataset_preprocessed.pkl'

# subset for early development
dataset_preprocessed_path = 'shared_resources/dataset_preprocessed_1000.pkl'

# unused
# dataset_tokenized_path = 'shared_resources/dataset_tokenized.npy'

# use only given fraction of dataset
dataset_fraction = 1.  # 0.1

# training subset fraction; validation and testing are divided equally among the rest
training_split = 0.8

embedding_dim = 32

""" Model """
model_weights_path = 'shared_resources/model_weights.pt'

model = {
    'input_size': embedding_dim,
    'hidden_size': 128,
    'output_size': 1
}

# Target labels
consistent = 0.0
inconsistent = 1.0

# Model hyperparameters
HPARS = {
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 1e-3,
    # 'weight_decay': 1e-3,

    'lr_scheduler_patience': 3,
    'lr_scheduler_min_lr': 1e-5,
    'lr_scheduler_factor': 0.5,

    'early_stopping_patience': 5,
}

twists = {
    'recombination': True,
    'flip_condition': True,
}

"""
# unused




"""
