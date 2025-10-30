from tqdm import tqdm
import os
import argparse
from utils import *
from loaders_location_verification import *
from llm_inference_location_verification import *
from location_verification_prompting import *


def verify_location_open(model, 
                         tokenizer, 
                         template,
                         image_path, 
                         candidate_location='', 
                         evidence_images=[], 
                         max_tokens=100, 
                         sleep=3):
    #Prompt for the task of location verification
    prompt= "You are a helpful assistant designed to support fact-checking. Your task is to verify whether a candidate location is accurate for an image.\n"
    if len(evidence_images)==0:
        prompt+= "You are given the image and the candidate location as input.\n"
    else:
        if 'map' in [ev[0] for ev in evidence_images]:
            if 'satellite' in [ev[0] for ev in evidence_images]:
                prompt+= "You are given the image, and a map and a satellite view of the candidate location as input.\n"
            else:
                prompt+= "You are given the image and a map of the candidate location as input.\n"

        else:
            prompt+= "You are given the image and a satellite view of the candidate location as input.\n"
    prompt+= "Answer only with true or false.\n"
    if candidate_location!='':
        prompt += f'Candidate location: {candidate_location}'
    if 'qwen2.5-vl' in template:
        output = generate_answer_qwen25vl(prompt, max_tokens, image_path, evidence_images, model, tokenizer)
    elif 'llama3.2' in template:
        output = generate_answer_llama32(prompt, max_tokens, image_path, evidence_images, model, tokenizer)
    else:
        #internvl2.5
        output = generate_answer_internvl2(prompt, max_tokens, image_path, evidence_images, model, tokenizer)
    return output




def main(template, zoom, use_satellite, use_map, use_candidate_str, pipeline_preds=[]):
    _, _, test = dataset_loader('data/M4FC.json', 'location_verification')


    #Load model
    if template in ['gpt4', 'gemini']:
        model = template
    else:
        model, tokenizer = load_model(template)

    os.makedirs('results', exist_ok=True)

    #Prepare the data
    candidate_locs = []
    coordinates = []
    ground_truth = []
    image_paths = []
    if len(pipeline_preds)==0:
        for t in range(len(test)):
                image_paths+= [test[t]['image_path'], test[t]['image_path'], test[t]['image_path']]
                candidate_locs.append(test[t]['location'])
                coordinates.append(test[t]['coordinates'])
                ground_truth.append('true')
                #negative locs
                candidate_locs += test[t]['negative_geolocations']
                coordinates += test[t]['negative_coordinates']
                ground_truth += ['false', 'false']
    else:
        #provide predicted locations and only generate for those
        for p in range(len(pipeline_preds)):
            image_paths.append(pipeline_preds[p]['image_path'])
            candidate_locs.append(pipeline_preds[p]['predicted_location'])
            coordinates.append(pipeline_preds[p]['coordinates'])
            ground_truth.append('')

    root_satellite = 'datastaticmap/satellite/'
    root_map = 'data/staticmap/osm/'


    output, usage_in, usage_out = [], [], []
    for c in tqdm(range(len(candidate_locs))):
        coo_1, coo_2 = coordinates[c].split('/')
        satellite_image_path = os.path.join(root_satellite, f"{coo_1}-{coo_2}-{zoom}.png")
        map_image_path = os.path.join(root_map, f"{coo_1}-{coo_2}-{zoom}.png")
        evidence_images = []
        if use_satellite:
            evidence_images.append(('satellite',satellite_image_path))
        if use_map:
            evidence_images.append(('map',map_image_path))
        if use_candidate_str:
            candidate_str = candidate_locs[c]
        else:
            candidate_str = ''
        if template in ['gpt4', 'gemini']:
            try:
                pred, tokens_in, tokens_out = verify_location_closed_source(template, image_paths[c], candidate_str, evidence_images, max_tokens=100, sleep=3)
            except:
                print('error')
                pred, tokens_in, tokens_out = False, 0, 0
        else:
            pred = verify_location_open(model, tokenizer, template, image_paths[c], candidate_str, evidence_images, max_tokens=100)
            tokens_in, tokens_out = 0, 0
        output.append(pred.lower().replace('\n','').replace('.',''))
        usage_in.append(tokens_in)
        usage_out.append(tokens_out)
    results = {'results':[]}
    for i in range(len(output)):
        results['results'].append({'image_path':image_paths[i], 
                        'candidate': candidate_locs[i],
                        'ground_truth': ground_truth[i],
                        'prediction': output[i]})
    template = args.model.split('/')[0]
    if len(preds)==0:
        output_path = f'results/location/location_verification_{template}'
    else:
        output_path = f'results/location/location_verification_{template}_pipeline'
    if use_satellite:
        output_path += '_satellite'
    if use_map:
        output_path += '_map'
    if use_candidate_str:
        output_path += '_str'
    output_path += f'_{zoom}.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt4") 
    parser.add_argument("--prediction_file", type=str, default="")
    args = parser.parse_args()


    preds = []

    main(args.model, 0, False, False, True, preds) #baseline --> just use the candidate location
    main(args.model, 15, True, False, False, preds)
    main(args.model, 15, False, True, False, preds)
    main(args.model, 15, True, True, False, preds)
    
    
