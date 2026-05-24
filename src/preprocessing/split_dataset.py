import os
import yaml
from sklearn.model_selection import train_test_split

input_path='data/processed/tokenized.jsonl'
output_dir='data/processed/'

def load_jsonl(filepath):
    data=[]
    with open(filepath,'r',encoding="utf8") as f:
        for line in f:
            if line.strip():
                data.append(line.strip())
    return data

def save_jsonl(data,filepath):
    with open(filepath,'w',encoding="utf8") as f:
        for line in data:
            f.write(line+"\n")

def split_and_save_dataset():
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy {input_path}.")
    
    document=load_jsonl(input_path)
    
    with open("config.yaml","r",encoding="utf8") as f:
        config=yaml.safe_load(f)
        
    TEST_SIZE=config["DATA"]["TEST_SIZE"]
    VAL_SIZE=config["DATA"]["VAL_SIZE"]
    
    train_docs,val_test_docs=train_test_split(document,test_size=TEST_SIZE+VAL_SIZE,random_state=42)
    
    test_ratio_adjusted = VAL_SIZE / (VAL_SIZE+TEST_SIZE)
    val_docs,test_docs=train_test_split(val_test_docs,test_size=test_ratio_adjusted,random_state=42)
    
    os.makedirs(output_dir, exist_ok=True)
    
    files_to_save = {
        f'{output_dir}train_data.jsonl': train_docs,
        f'{output_dir}val_data.jsonl': val_docs,
        f'{output_dir}test_data.jsonl': test_docs
    }
    
    for filename,data in files_to_save.items():
        save_jsonl(data,filename)
        print(f"Save {filename}: {len(data):,} rows")
        
if __name__ == "__main__":
    split_and_save_dataset()