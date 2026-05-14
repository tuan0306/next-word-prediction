import os
import requests
import json
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
]

def load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf8') as f:
                return set(json.load(f))
        except json.decoder.JSONDecodeError:
            return set()
    return set()

def save_checkpoint(done_urls,path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.',exist_ok=True)
    
    with open(path,'w',encoding='utf8') as f:
        json.dump(list(done_urls),f,indent=4,ensure_ascii=False)
        
def append_jsonl(record, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.',exist_ok=True)
    
    with open(path,'a',encoding='utf8') as f:
        json_string=json.dumps(record,ensure_ascii=False)
        f.write(json_string+'\n')
        
def make_session(retries=3):
    session=requests.Session()
    session.headers.update({
        "User-Agent":random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,...",
        "Accept-Language":"vi-VN,vi;q=0.9,en-US;q=0.8",
    })
    retry=Retry(total=retries,backoff_factor=1,status_forcelist=[429,500,502,503,504])
    adapter=HTTPAdapter(max_retries=retry)
    session.mount("http://",adapter=adapter)
    session.mount("https://",adapter=adapter)
    return session
    