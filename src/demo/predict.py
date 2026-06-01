import numpy as np
import tensorflow as tf
from src.preprocessing.clean_text import clean_text
from src.preprocessing.tokenize import tokenize_text

def predict_topk(model, text, word2idx, idx2word, max_length, k=3):
    cleaned_text=clean_text(text)
    tokens=tokenize_text(cleaned_text).split()
    unk_id=word2idx.get("<UNK>",1)
    ids=[word2idx.get(word,unk_id) for word in tokens]
    ids = ids[-max_length:]
    pad_length=max_length-len(ids)
    if pad_length>0:
        ids =  ids+([0] * pad_length)
    x=np.array([ids],dtype=np.int32)
    preds=model.predict(x,verbose=0)
    logits=preds[0]
    probs=tf.nn.softmax(logits).numpy()
    top_k_indices = np.argsort(probs)[-k:][::-1]
    results = []
    for idx in top_k_indices:
        word = idx2word.get(str(idx), "<UNK>")
        prob = probs[idx]
        results.append((word, float(prob)))
    return results