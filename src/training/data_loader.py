import json
import numpy as np
import tensorflow as tf
import yaml

pad_sequences=tf.keras.preprocessing.sequences.pad_sequences

def load_vocab(filepath):
    with open(filepath,'r',encoding="utf8") as f:
        return json.load(f)
    

    
def sequence_generator(filepath,vocab,max_length):
    unk_id=vocab.get("<UNK>",1)
        
    with open(filepath,'r',encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            
            line=line.strip()
            
            try:
                doc=json.loads(line)
                
                content=doc.get("title","") + " " + doc.get("content","")
                tokens=content.split()
                
                if len(tokens)<5:
                    continue
                
                tokens=vocab.get("<BOS>",0)+tokens+vocab.get("<EOS>",3)
                
                token_list=[vocab.get(word,unk_id) for word in tokens]
                
                for i in range(1,len(token_list)):
                    n_gram=token_list[:i+1]
                    
                    padded=pad_sequences(
                        [n_gram],
                        maxlen=max_length+1,
                        padding='pre',
                        truncating='pre'
                    )[0]
                    
                    X=padded[:-1]
                    y=padded[-1]
                    
                    yield X,y
                    
            except json.JSONDecodeError:
                continue
            
def build_streaming_dataset(filepath,vocab_path,batch_size=128,max_length=20,shuffle_buffer=10000, is_training=False):
    vocab=load_vocab(vocab_path)
    
    dataset=tf.data.Dataset.from_generator(
        generator=lambda: sequence_generator(filepath,vocab,max_length),
        output_signature=(
            tf.TensorSpec(shape=(max_length,),dtype=tf.int32),
            tf.TensorSpec(shape=(),dtype=tf.int32)
        )
    )
    
    if is_training:
        dataset=dataset.shuffle(buffer_size=shuffle_buffer,reshuffle_each_iteration=True)
        
    dataset=dataset.batch(batch_size)
    dataset=dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
