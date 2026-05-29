import os
import sys
import json
import argparse
import math
import numpy as np
import tensorflow as tf
import yaml

from src.models.rnn_model import RNNModel
from src.models.lstm_model import LSTMModel
from src.training.data_loader import build_streaming_dataset

def main():
    with open('config.yaml','r',encoding="utf8") as f:
        config=yaml.safe_load(f)
        
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['rnn','lstm'])
    parser.add_argument('--batch_size',type=int,default=config["TRAINING"]["BATCH_SIZE"])
    args=parser.parse_args()
    
    vocab_path='data/processed/vocab.json'
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_size=vocab["vocab_size"]
        
    test_ds=build_streaming_dataset('data/processed/test_data.jsonl',vocab_path,
                                    batch_size=args.batch_size,max_length=config["TRAINING"]["MAX_LENGTH"],
                                    is_finite=True)
    
    model_params={
        'vocab_size':vocab_size,
        'embed_dim':config["MODEL"]["EMBEDDING_DIM"],
        'hidden_size': config["MODEL"]["HIDDEN_SIZE"],
        'num_layers': config["MODEL"]["NUM_LAYERS"],
        'dropout': config["MODEL"]["DROPOUT"]
    }
    
    if args.model=='rnn':
        model=RNNModel(**model_params)
        checkpoint_path='checkpoints/rnn_best.weights.h5'
    else:
        model=LSTMModel(**model_params)
        checkpoint_path='checkpoints/lstm_best.weights.h5'
    
    dummy_input = tf.zeros((1, config["TRAINING"]["MAX_LENGTH"]), dtype=tf.int32)
    model(dummy_input)
    model.load_weights(checkpoint_path)
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1,name="top1_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3,name="top3_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,name="top5_acc")
        ]
    )
    
    eval_results=model.evaluate(test_ds,verbose=1,return_dict=True)
    
    loss=eval_results['loss']
    top1=eval_results['top1_acc']
    top3=eval_results['top3_acc']
    top5=eval_results['top5_acc']
    
    try:
        perplexity=math.exp(loss)
    except OverflowError:
        perplexity = float('inf')
        
    print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH {args.model.upper()}")
    print(f"Loss (Cross-Entropy) : {loss:.4f}")
    print(f"Perplexity           : {perplexity:.2f}")
    print(f"Top-1 Accuracy       : {top1 * 100:.2f}%")
    print(f"Top-3 Accuracy       : {top3 * 100:.2f}%")
    print(f"Top-5 Accuracy       : {top5 * 100:.2f}%")
    
    report_dir='report'
    os.makedirs(report_dir,exist_ok=True)
    results_file=f'{report_dir}/results.json'
    
    results_dict={}
    if os.path.exists(results_file):
        with open(results_file,'r',encoding="utf8") as f:
            try:
                results_dict=json.load(f)
            except json.JSONDecodeError:
                results_dict = {}
                
    results_dict[args.model]={
        'val_loss': round(loss, 4),
        'perplexity':round(perplexity,2) if perplexity != float('inf') else "INF",
        'top_1_accuracy': round(top1, 4),
        'top_3_accuracy': round(top3, 4),
        'top_5_accuracy': round(top5, 4)
    }
    
    with open(results_file,'w',encoding='utf8') as f:
        json.dump(results_dict,f,ensure_ascii=False,indent=4)
        
if __name__ == "__main__":
    main()