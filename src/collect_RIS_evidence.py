import os 
from tqdm import tqdm
from _utils import *
from scrape_utils import *
import argparse



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Collect evidence using Google Reverse Image Search.')
    parser.add_argument('--evidence_urls', type=str, default='data/retrieval_results/evidence_urls.json',
                        help='Path to the list of evidence URLs to scrape. Needs to be a valid file if collect_google is set to 0.')
    parser.add_argument('--scrape_with_trafilatura', type=int, default=1, 
                        help='Whether to scrape the evidence URLs with trafilatura. If 0, it is assumed that a file containing the scraped webpages already exists.') 
    parser.add_argument('--trafilatura_path', type=str, default='data/retrieval_results/trafilatura_data.json',
                        help='The json file to store the scraped trafilatura  content as a json file.')
    parser.add_argument('--apply_filtering', type=int, default=0,
                        help='If 1, remove evidence published after the source FC article. Not needed if using the default evidence set')
    parser.add_argument('--json_path', type=str, default='data/retrieval_results/evidence.json',
                        help='The json file to store the text evidence as a json file.')
    parser.add_argument('--max_results', type=int, default=20,
                        help='The maximum number of web-pages to collect with the web detection API.') 
    parser.add_argument('--sleep', type=int, default=3,
                        help='The waiting time between two web detection API calls') 
    

    args = parser.parse_args()

    #Create directories if they do not exist yet
    if not 'retrieval_results'  in os.listdir('data/'):
        os.mkdir('data/retrieval_results/')

    selected_data = [d for d in load_json(args.evidence_urls) if d['image path'].split('/')[-1] in os.listdir('data/images/')]
        
    urls = [d['raw_url'] for d in selected_data]
    images = [d['image_urls'] for d in selected_data]

    if args.scrape_with_trafilatura:
        #Collect results with Trafilatura
        output = []
        for u in tqdm(range(len(urls))):
            output.append(extract_info_trafilatura(urls[u],images[u]))
            #Only store in json file every 50 evidence
            if u%50==0:
                save_result(output,args.trafilatura_path) 
                output = []
    
    #Save all results in a Pandas Dataframe
    evidence_trafilatura = load_json(args.trafilatura_path)
    dataset = load_json('data/M4FC.json')
    evidence = merge_data(evidence_trafilatura, selected_data, dataset, apply_filtering=args.apply_filtering).fillna('').to_dict(orient='records')
    # Save the list of dictionaries as a JSON file
    with open(args.json_path, 'w') as file:
        json.dump(evidence, file, indent=4)
