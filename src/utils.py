from PIL import Image
import io, os, base64, json


def load_json(file_path):
    '''
    Load json file
    '''
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

def concatenate_entry(d):
    '''
    For all keys in a dictionary, if a value is a list, concatenate it.
    '''
    for key, value in d.items():
        if isinstance(value, list):  
            d[key] = ';'.join(map(str, value))  # Convert list to a string separated by ';'
    return d


def append_to_json(file_path, data):
    '''
    Append a dict or a list of dicts to a JSON file.
    '''
    try:
        if not os.path.exists(file_path):
            # Create an empty JSON file with an empty list if it does not exist yet
            with open(file_path, 'w') as file:
                json.dump([], file)
        #Open the existing file
        with open(file_path, 'r+') as file:
            file_data = json.load(file)
            if type(data)==list:
                for d in data:
                    if type(d)==dict:
                        file_data.append(concatenate_entry(d))
            else:
                file_data.append(concatenate_entry(data))
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")


def save_result(output,json_file_path):
    '''
    Save output results to a JSON file.
    '''
    if type(output)==str:
        output = json.loads(output)
        append_to_json(json_file_path, output)
    else:
        append_to_json(json_file_path, output)


def dataset_loader(task='verdict_prediction', 
                   balanced=False,
                   multilingual=False):
    '''
    Get train, dev, test data for a given task
    Params:
        task (str): the selected task
        balanced (bool): only relevant for verdict prediction, if True uses the balanced setting
        multilingual (bool): only relevant for verdict prediction, if True uses the multilingual setting
    '''
    data = load_json('data/M4FC.json')
    #take the relevant subset of the data
    if task=='claim_normalization':
        subset = [d for d in data if d['task_claim_normalization']==True]
    elif task=='claimant_motivation':
        subset = [d for d in data if d['task_claimant_intent']==True]
    elif task=='fake_detection':
        subset = [d for d in data if d['task_fake_detection']==True]
    elif task=='image_contextualization':
        subset = [d for d in data if d['task_image_contextualization']==True]
    elif task=='location_verification':
        subset = [d for d in data if d['task_location_verification']==True]
    elif task=='verdict_prediction':
        subset = [d for d in data if  d['task_verdict_prediction']==True]
        #Adjust claims based on verdict prediction setting
        for s in range(len(subset)):
            if balanced and multilingual:
                #Balanced multilingual setting
                    if subset[s]['use_true_caption']:
                        subset[s]['claim'] = subset[s]['multilingual_true_caption']  if subset[s]['multilingual_true_caption']!='not enough information' else subset[s]['true_caption']
                    else:
                        subset[s]['claim']= subset[s]['multilingual_claim']  if subset[s]['multilingual_claim']!='not enough information' else subset[s]['claim']
            elif balanced:
                if subset[s]['use_true_caption']:
                        subset[s]['claim'] = subset[s]['true_caption']
            elif multilingual:
                if subset[s]['multilingual_claim']!= 'not enough information':
                    subset[s]['claim'] = subset[s]['multilingual_claim']
            else:
                pass   
    elif task=='fc_pipeline':
        subset = data
    else:
        print('Invalid task name provided')
    #train, dev, test splits
    train = [d for d in subset if d['split']=='train']
    dev = [d for d in subset if d['split']=='dev']
    test = [d for d in subset if d['split']=='test']
    return train, dev, test
            
            
def load_image(image_path):
    """Load an image from the given path and return it as a base64-encoded string."""
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format=img.format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")