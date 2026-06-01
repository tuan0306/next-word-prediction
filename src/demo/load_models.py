import json
from pathlib import Path
import tensorflow as tf
import yaml
from src.models.rnn_model import RNNModel
from src.models.lstm_model import LSTMModel

def load_models():
    project_root = Path(__file__).resolve().parents[2]
    vocab_path = project_root / 'data' / 'processed' / 'vocab.json'
    with open(vocab_path,'r',encoding='utf8') as f:
        vocab=json.load(f)

    with open(project_root / 'config.yaml','r',encoding='utf8') as f:
        config=yaml.safe_load(f)
        
    vocab_size=vocab['vocab_size']
    model_params={
        'vocab_size':vocab_size,
        'embed_dim':config["MODEL"]["EMBEDDING_DIM"],
        'hidden_size': config["MODEL"]["HIDDEN_SIZE"],
        'num_layers': config["MODEL"]["NUM_LAYERS"],
        'dropout': config["MODEL"]["DROPOUT"]
    }
    
    rnn_model=RNNModel(**model_params)
    lstm_model=LSTMModel(**model_params)

    rnn_checkpoints = project_root / 'checkpoints' / 'rnn_best.weights.h5'
    lstm_checkpoints = project_root / 'checkpoints' / 'lstm_best.weights.h5'

    dummy_input = tf.zeros((1, config["TRAINING"]["MAX_LENGTH"]), dtype=tf.int32)
    rnn_model(dummy_input)
    lstm_model(dummy_input)

    rnn_model.load_weights(rnn_checkpoints)
    lstm_model.load_weights(lstm_checkpoints)
    
    return rnn_model,lstm_model,vocab, config["TRAINING"]["MAX_LENGTH"]