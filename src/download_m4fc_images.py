import argparse
import os
from tqdm import tqdm
import time
from scrape_utils import *
from utils import *


def download_image_dataset(file_path, sleep=2):
    '''
    Donwload all images in 8Pils and store them in a common images folder
    '''
    #Load the dataset
    dataset = load_json(file_path)
    already_downloaded_images = set([f'data/images/{im}' for im in os.listdir('data/images')])

    for i in tqdm(range(len(dataset))):
        #Download all images of the dataset
        file_path = os.path.join('data', dataset[i]['image_path']).replace('\\', '/')
        url = dataset[i]['image_url']
        if file_path not in already_downloaded_images:
            #The image has not been downloaded yet
            success = scrape_image(url, file_path)
            if not success and 'wayback_image_url' in dataset[i].keys():
                wayback_url = dataset[i]['wayback_image_url']
                success =scrape_image(wayback_url, file_path)
            if not success:
                print(url)
            time.sleep(sleep)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Download images based on provided URLs.')
    parser.add_argument('--sleep', type=int, default=5,
                        help='Waiting time in seconds between scraping two images.')

    args = parser.parse_args()
    os.makedirs('data/images',exist_ok=True)
    download_image_dataset('data/M4FC.json', args.sleep)