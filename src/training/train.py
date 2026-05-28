import os
import json
import yaml
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from src.training.data_loader import build_streaming_dataset
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import RNNModel

def plot_and_save_history(history, model_name):
    os.makedirs('report/figures', exist_ok=True)
    
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f'{model_name.upper()} - Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name.upper()} - Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(f'report/figures/{model_name}_training_curve.png')
    
def main():
    with open('config.yaml','r',encoding="utf8") as f:
        config=yaml.safe_load(f)
        
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=str,required=True,choices=['rnn','lstm'])
    parser.add_argument('--epochs',type=int,default=config["TRAINING"]["EPOCHS"])
    parser.add_argument('--lr',type=float,default=config["TRAINING"]["LEARNING_RATE"])
    parser.add_argument('--batch_size',type=int,default=config["TRAINING"]["BATCH_SIZE"])
    parser.add_argument('--max_length',type=int,default=config["TRAINING"]["MAX_LENGTH"])
    args=parser.parse_args()
    
    vocab_path='data/processed/vocab.json'
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    vocab_size=vocab["vocab_size"]
        
    train_ds=build_streaming_dataset('data/processed/train_data.jsonl',vocab_path,
                                     batch_size=args.batch_size,max_length=args.max_length,
                                     is_training=True)
    
    val_ds=build_streaming_dataset('data/processed/val_data.jsonl',vocab_path,
                                     batch_size=args.batch_size,max_length=args.max_length,
                                     is_training=False)
    
    model_params={
        'vocab_size':vocab_size,
        'embed_dim':config["MODEL"]["EMBEDDING_DIM"],
        'hidden_size': config["MODEL"]["HIDDEN_SIZE"],
        'num_layers': config["MODEL"]["NUM_LAYERS"],
        'dropout': config["MODEL"]["DROPOUT"]
    }
    
    if args.model=='rnn':
        model=RNNModel(**model_params)
        
    else:
        model=LSTMModel(**model_params)
        
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr,clipnorm=1.0)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    for x_batch,y_batch in train_ds.take(1):
        model(x_batch,training=False)
    
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f"checkpoints/{args.model}_best.weights.h5"
    
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history=model.fit(
        train_ds,validation_data=val_ds,epochs=args.epochs,callbacks=callbacks,
        steps_per_epoch=500,validation_steps=200
    )
    
    plot_and_save_history(history, args.model)
    
if __name__=="__main__":
    main()