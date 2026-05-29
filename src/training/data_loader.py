import json
import numpy as np
import tensorflow as tf
import yaml

def load_vocab(filepath):
    with open(filepath,'r',encoding="utf8") as f:
        vocab_data = json.load(f)

    return vocab_data["word2idx"]
    

    
def sequence_generator(filepath,vocab,max_length):
    unk_id=vocab.get("<UNK>",1)
    
    while True:
        with open(filepath,'r',encoding="utf8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                line=line.strip()
                
                try:
                    doc=json.loads(line)
                    
                    content=doc.get("content","")
                    tokens=content.split()
                    
                    if len(tokens)<5:
                        continue
                    
                    tokens=["<BOS>"]+tokens+["<EOS>"]
                    
                    token_list=[vocab.get(word,unk_id) for word in tokens]
                    
                    for i in range(1,len(token_list),2):
                        n_gram=token_list[:i+1]
                        
                        X=n_gram[:-1].copy()
                        y=n_gram[-1]
                        
                        if len(X)>max_length:
                            X=X[-max_length:]
                        
                        pad_length=max_length-len(X)
                        
                        if pad_length>0:
                            X=X+[vocab.get("<PAD>",0)]*pad_length
                        
                        yield X,y
                        
                except json.JSONDecodeError:
                    continue
                
def finite_eval_generator(filepath,vocab,max_length):
    unk_id=vocab.get("<UNK>",1)
    
    with open(filepath,'r',encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            
            line=line.strip()
            
            try:
                doc=json.loads(line)
                
                content=doc.get("content","")
                tokens=content.split()
                
                if len(tokens)<5:
                    continue
                
                tokens=["<BOS>"]+tokens+["<EOS>"]
                
                token_list=[vocab.get(word,unk_id) for word in tokens]
                
                for i in range(1,len(token_list),2):
                    n_gram=token_list[:i+1]
                    
                    X=n_gram[:-1].copy()
                    y=n_gram[-1]
                    
                    if len(X)>max_length:
                        X=X[-max_length:]
                    
                    pad_length=max_length-len(X)
                    
                    if pad_length>0:
                        X=X+[vocab.get("<PAD>",0)]*pad_length
                    
                    yield X,y
                    
            except json.JSONDecodeError:
                continue
            
def build_streaming_dataset(filepath,vocab_path,batch_size=128,max_length=20,shuffle_buffer=10000, is_training=False, is_finite=False):
    vocab=load_vocab(vocab_path)
    
    if not is_finite:
        dataset=tf.data.Dataset.from_generator(
            generator=lambda: sequence_generator(filepath,vocab,max_length),
            output_signature=(
                tf.TensorSpec(shape=(max_length,),dtype=tf.int32),
                tf.TensorSpec(shape=(),dtype=tf.int32)
            )
        )
        
    else:
        dataset=tf.data.Dataset.from_generator(
            generator=lambda: finite_eval_generator(filepath,vocab,max_length),
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
