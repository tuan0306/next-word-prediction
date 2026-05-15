import os
import re
import unicodedata
import json
from tqdm import tqdm
import glob

output_dir='data/processed/'

output_file='data/processed/cleaned.jsonl'

input_dir='data/raw/'

def clean_text(text):
    if not isinstance(text,str):
        return ""
    
    text=unicodedata.normalize("NFC",text)
    text=text.lower()
    
    # Xoa html tags
    text=re.sub(r'<[^>]+>',' ',text)
    
    # Xoa urls
    text=re.sub(r'https?://\S+|www\.\S+',' ',text)
    
    # Xoa email
    text=re.sub(r'\b[a-z0-9+-_%.]+@[a-z0-0.-]\.[a-z]{2,}\b',' ',text)
    
    # Xoa so dien thoai
    text=re.sub(r'\b(0|\+84)\d{9}\b',' ',text)
    
    # Xoa ky tu dac biet
    text=re.sub(r'[^\w\s.,!?]+',' ',text)
    
    # Chuan hoa khoang trang
    text=re.sub(r'\s+',' ',text).strip()
    
    return text

def main():
    os.makedirs(output_dir,exist_ok=True)
    
    raw_files=glob.glob(f"{input_dir}*.jsonl")
    
    if not raw_files:
        print("Khong tim thay file jsonl")
        return
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    total_processed = 0
    total_discarded = 0
    
    for file_path in raw_files:
        with open(file_path,'r',encoding='utf8') as infile, \
            open(output_file,'a',encoding='utf8') as outfile:
                lines=infile.readlines()
                
                for line in tqdm(lines,desc='Process of cleaning text'):
                    if not line.strip():
                        continue
                    
                    try:
                        data=json.loads(line)
                        cleaned_title=clean_text(data.get("title",""))
                        cleaned_content=clean_text(data.get("content",""))
                        if len(cleaned_content)>20:
                            cleaned_data={
                                "url":data.get("url",""),
                                "title":cleaned_title,
                                "content":cleaned_content
                            }
                            outfile.write(json.dumps(cleaned_data,ensure_ascii=False)+'\n')
                            total_processed+=1
                        else:
                            total_discarded+=1
                    
                    except json.JSONDecodeError:
                        continue
    print(f"Number of processed articles: {total_processed}")
    print(f"Number of discarded articles: {total_discarded}")
    
if __name__=="__main__":
    main()