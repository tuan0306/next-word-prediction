import os
import time
import wikipediaapi
from tqdm import tqdm

from utils import load_checkpoint,save_checkpoint,append_jsonl

checkpoint_file="src/1_scraping/checkpoints/scraped_wikipedia.json"
output_file="data/raw/wikipedia.jsonl"

USER_AGENT = 'NWP-Thesis/1.0 (https://github.com/tuan0306/NWP-Thesis)'

wiki=wikipediaapi.Wikipedia(
    language='vi', 
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent=USER_AGENT
)

CATEGORIES=[
    'Thể loại:Khoa học',
    'Thể loại:Lịch sử',
    'Thể loại:Công nghệ',
    'Thể loại:Địa lý',
    'Thể loại:Xã hội',
    'Thể loại:Thể thao',
]

def get_all_pages_in_category(category_members,max_depth=1,current_depth=0):
    page_titles=[]
    
    for title,page in category_members.items():
        if page.ns==wikipediaapi.Namespace.MAIN:
            page_titles.append(title)
        
        elif page.ns==wikipediaapi.Namespace.CATEGORY and current_depth<max_depth:
            pages=get_all_pages_in_category(page.categorymembers,max_depth,current_depth+1)
            page_titles.extend(pages)
    return page_titles

def scrape_article(page_title):
    try:
        page=wiki.page(page_title)
        if not page.exists() or not page.text.strip():
            return None
        return {
            "url":page.fullurl,
            "title":page.title,
            "content":page.text
        }
    except Exception as e:
        return None
    
def main():
    scraped_titles=load_checkpoint(checkpoint_file)
    print(f"Number of scraped articles: {len(scraped_titles)}")
    
    all_target_titles=[]
    
    for cat_tag in CATEGORIES:
        page=wiki.page(cat_tag)
        
        if not page.exists():
            continue
        
        titles=get_all_pages_in_category(page.categorymembers,max_depth=1)
        all_target_titles.extend(titles)
        
    all_target_titles=list(set(all_target_titles))
    
    new_titles=[title for title in all_target_titles if title not in scraped_titles]
    
    print(f"Total of all found articles: {len(all_target_titles)} link.")
    print(f"Number of new articles need to crawl: {len(new_titles)}\n")
    
    for title in tqdm(new_titles,desc="Process of crawling"):
        article=scrape_article(title)
        if article:
            append_jsonl(article,output_file)
            scraped_titles.add(title)
            save_checkpoint(scraped_titles,checkpoint_file)
        time.sleep(0.1)

if __name__=="__main__":
    main()