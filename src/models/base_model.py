import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

class BaseNWPModel(tf.keras.Model,ABC):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    @abstractmethod
    def call(self,inputs,training=False):
        pass
    
    def predict_top_k(self,x,k=3):
        logits=self(x,training=False)
        if len(logits.shape)==3:
            logits=logits[:,-1,:]
        
        probs=tf.nn.softmax(logits,axis=-1)
        top_k_probs,top_k_indices=tf.math.top_k(probs,k=k)
        return top_k_probs.numpy(),top_k_indices.numpy()
    
    def get_param_count(self):
        trainable=sum([tf.keras.backend.count_params(w) for w in self.trainable_weights])
        non_trainable=sum([tf.keras.backend.count_params(w) for w in self.non_trainable_weights])
        total=trainable+non_trainable
        
        print(f"Total params: {total}")
        print(f"Trainable params: {trainable}")
        print(f"Non-trainable params: {non_trainable}")
        
    def save_model_weights(self,filepath):
        self.save_weights(filepath)
    
    def load_model_weights(self,filepath):
        self.load_weights(filepath)
        