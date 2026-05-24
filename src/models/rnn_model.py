import tensorflow as tf
from .base_model import BaseNWPModel

class RNNModel(BaseNWPModel):
    def __init__(self,vocab_size,embed_dim=128,hidden_size=256,num_layers=2,dropout=0.3,**kwargs):
        super().__init__(**kwargs)
        
        self.embedding=tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True
        )
        
        self.dropout1=tf.keras.layers.Dropout(rate=dropout)
        
        self.rnn_layers=[]
        
        for i in range(num_layers):
            is_last_layer=(i==num_layers-1)
            self.rnn_layers.append(
                tf.keras.layers.SimpleRNN(
                    units=hidden_size,
                    return_sequences=not is_last_layer
                )
            )
        
        self.dropout2=tf.keras.layers.Dropout(rate=dropout)
        
        self.dense=tf.keras.layers.Dense(units=vocab_size)
        
    def call(self,inputs,training=False):
        x=self.embedding(inputs)
        x=self.dropout1(x,training=training)
        for rnn in self.rnn_layers:
            x=rnn(x,training=training)
        x=self.dropout2(x,training=training)
        logits=self.dense(x)
        return logits