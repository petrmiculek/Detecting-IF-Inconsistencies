# Global config module

# Tokenization
model_input_len = 256

# Paths
# Word Embedding
fasttext_path = '../shared_resources/pretrained_fasttext/embed_if_32.mdl'

dataset_tokenized_path = '../shared_resources/dataset_tokenized.npy'
dataset_preprocessed_path = '../shared_resources/dataset_preprocessed.pkl'

model_weights_path = '../shared_resources/model_weights.pt'

embedding_dim = 32


# Target labels
consistent = 0.0
inconsistent = 1.0

# Model hyperparameters
HYPERPARAMETERS = {
    'batch_size': 4,
    'epochs': 10,
    'learning_rate': 1e-3,
    'weight_decay': 1e-3,

    'lr_scheduler_patience': 5,
    'lr_scheduler_min_lr': 1e-6,
    'lr_scheduler_factor': 0.5,

    'early_stopping_patience': 10,
}
