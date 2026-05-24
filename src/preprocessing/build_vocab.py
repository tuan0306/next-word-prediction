import yaml
import os
import json
from collections import Counter
from tqdm import tqdm

input_file='data/processed/tokenized.jsonl'
output_file='data/processed/vocab.json'

def main():
    if not os.path.exists(input_file):
        print("Chua co file input")
        return
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open("config.yaml","r",encoding="utf8") as f:
        config=yaml.safe_load(f)
        
    VOCAB_SIZE=config["VOCAB"]["VOCAB_SIZE"]
    SPECIAL_TOKENS=config["VOCAB"]["SPECIAL_TOKENS"]
    
    word_counter=Counter()
    total_articles = 0
    
    with open(input_file,'r',encoding="utf8") as infile:
        for line in tqdm(infile,desc="Process of counting word frequency"):
            if not line.strip():
                continue
            
            try:
                data=json.loads(line)
                text=data.get("title","")+ " " + data.get("content","")
                tokens=text.split()
                word_counter.update(tokens)
                total_articles+=1
            except json.JSONDecodeError:
                continue
    
    print(f"Number of processed articles: {total_articles}")
    print(f"Number of unique words: {len(word_counter)}")
    
    most_common_words=[word for word,count in word_counter.most_common(VOCAB_SIZE)]
    vocab_words=SPECIAL_TOKENS+most_common_words
    
    word2idx={word: idx for idx,word in enumerate(vocab_words)}
    idx2word={idx: word for idx,word in enumerate(vocab_words)}
    
    vocab_data={
        'word2idx':word2idx,
        'idx2word':idx2word,
        'vocab_size':len(vocab_words)
    }
    
    with open(output_file,'w',encoding="utf8") as outfile:
        json.dump(vocab_data, outfile, indent=2, ensure_ascii=False)
        
if __name__=="__main__":
    main()