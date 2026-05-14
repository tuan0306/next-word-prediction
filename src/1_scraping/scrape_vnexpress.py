import json
import os
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
import time

from utils import load_checkpoint,save_checkpoint,make_session,append_jsonl

checkpoint_file='src/1_scraping/checkpoints/scraped_vnexpress.json'
data_output_file='data/raw/vnexpress.jsonl'

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

CATEGORIES = [
  'https://vnexpress.net/khoa-hoc-cong-nghe',
  'https://vnexpress.net/doi-song',
  'https://vnexpress.net/the-thao',
  'https://vnexpress.net/giai-tri',
  'https://vnexpress.net/giao-duc',
  'https://vnexpress.net/phap-luat'
]

session=make_session()

def scrape_category(base_url):
    article_links=[]
    page=1
    while True:
        if page==1:
            url=base_url
        else:
            url=base_url+f"-p{page}"
        try:
            response=session.get(url,timeout=10)
            if response.status_code!=200:
                break
            soup=BeautifulSoup(response.content,'html.parser')
            page_links=[]
            
            articles=soup.select(".title-news a")
            
            for a_tag in articles:
                link=a_tag.get("href")
                if link and link.startswith("https://vnexpress.net/") and "podcast" not in link:
                    page_links.append(link)
                    
            if not page_links:
                break
            
            article_links.extend(page_links)
            page+=1
            time.sleep(0.5)
        
        except Exception as e:
            print(f"Lỗi mạng khi quét {url}: {e}")
            break
                    
    return list(set(article_links))
    
def scrape_article(url):
    try:
        response=session.get(url,timeout=10)
        if response.status_code != 200:
            return None
        soup=BeautifulSoup(response.content,'html.parser')
        
        title_tag=soup.select_one("h1.title-detail")
        title=title_tag.text.strip() if title_tag else ""
        
        paragraphs=soup.select("p.Normal")
        content="\n".join([p.text.strip() for p in paragraphs])
        if not title or not content:
            return None
        return {
            "url":url,
            "title":title,
            "content":content
        }
    except Exception as e:
        return None

def main():
    scraped_urls=load_checkpoint(checkpoint_file)
    print(f"Number of scraped articles: {len(scraped_urls)}")
    
    all_found_links=[]
    for cat_url in CATEGORIES:
        links=scrape_category(cat_url)
        all_found_links.extend(links)
        time.sleep(1)
    
    all_found_links=list(set(all_found_links))
    new_urls=[url for url  in all_found_links if url not in scraped_urls]
    
    print(f"Total of all found urls: {len(all_found_links)} link.")
    print(f"Number of new urls need to crawl: {len(new_urls)}\n")
    
    for url in tqdm(new_urls,desc="Process of crawling"):
        artical_data=scrape_article(url)
        if artical_data:
            append_jsonl(artical_data,data_output_file)
            scraped_urls.add(url)
            save_checkpoint(scraped_urls,checkpoint_file)
        time.sleep(1)
        
if __name__=="__main__":
    main()