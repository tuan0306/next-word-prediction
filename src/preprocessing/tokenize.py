import json
import os
from underthesea import word_tokenize
from tqdm import tqdm

input_file='data/processed/cleaned.jsonl'
output_file='data/processed/tokenized.jsonl'

def tokenize_text(text):
    if not text:
        return ""
    
    return word_tokenize(text,format="text")

def main():
    if not os.path.exists(input_file):
        print("Chua co file du lieu dau vao")
        return 
    
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    total_processed = 0
    
    with open(input_file,'r',encoding="utf8") as infile, \
        open(output_file,'a',encoding="utf8") as outfile:
            lines=infile.readlines()
            for line in tqdm(lines,desc="Process of tokenizing"):
                if not line.strip():
                    continue
                
                try:
                    data=json.loads(line)
                    tokenized_title=tokenize_text(data.get("title",""))
                    tokenized_content=tokenize_text(data.get("content",""))
                    tokenized_data={
                        "url":data.get("url",""),
                        "title":tokenized_title,
                        "content":tokenized_content
                    }
                    outfile.write(json.dumps(tokenized_data,ensure_ascii=False)+"\n")
                    total_processed+=1
                except Exception as e:
                    continue
    print(f"Number of processed articles: {total_processed}")
    
if __name__=="__main__":
    main()