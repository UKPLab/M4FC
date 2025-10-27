import argparse
from tqdm import tqdm
import time
from waybackpy import WaybackMachineSaveAPI, WaybackMachineCDXServerAPI
from scrape_utils import *
from utils import *

def save_url(url):
    user_agent = "Mozilla/5.0 (Windows NT 5.1; rv:40.0) Gecko/20100101 Firefox/40.0"
    save_api = WaybackMachineSaveAPI(url, user_agent,max_tries=1)
    wayback_url = save_api.save()
    return wayback_url

def check_availability_url(url):
    user_agent = "Mozilla/5.0 (Windows NT 5.1; rv:40.0) Gecko/20100101 Firefox/40.0"
    cdx_api = WaybackMachineCDXServerAPI(url, user_agent,max_tries=1)
    newest = cdx_api.newest()
    url = newest.archive_url
    return url


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Download images based on provided URLs.')
    parser.add_argument('--sleep', type=int, default=10,
                        help='Waiting time in seconds between scraping two images.')
    args = parser.parse_args()
    data = load_json('data/M4FC.json')
    counter = 0
    for d in tqdm(range(len(data)-1,0,-1)):
        #start with snopes only and if not already done
        if 'wayback_image_url' not in data[d].keys() and 'web.archive.org/' not in data[d]['image_url']:
            url = data[d]['image_url']
            try:
                wayback_url = check_availability_url(url)
                time.sleep(args.sleep)
            except:
                #save new url
                try:
                    if 'format:webp' in url and 'miro.medium' in url:
                        url = url.replace('format:webp', 'format:png')
                    wayback_url = save_url(url)
                    time.sleep(args.sleep)
                except Exception as e:
                    print(e)
                    print(url)
                    wayback_url = None
            if wayback_url:
                counter += 1
                data[d]['wayback_image_url'] = wayback_url

            if counter==10:
                with open('data/M4FC.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                counter = 0
    with open('data/M4FC.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)