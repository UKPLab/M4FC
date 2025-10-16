import Levenshtein as lev
from dateutil.tz import tzutc
from dateutil import parser
import requests
from trafilatura import bare_extraction
from bs4 import BeautifulSoup as bs
import pandas as pd
import requests as rq
from io import BytesIO
from utils import *

def scrape_image(url, file_path):
    '''
    Scrape an image given its url and store it locally as a png file.
    '''
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        req = rq.get(url, stream=True, timeout=(10,10), headers=headers)
    except:
        return None
    if req.status_code!= 200:
        print(req)
    if req.status_code == 200 and ('image' in req.headers.get('Content-Type', '') or 'factchecklab.org' in url) :
        image_content = req.content
        try:
            image = Image.open(BytesIO(image_content))
            image.verify()
            with Image.open(BytesIO(image_content)) as img:
                img.save(file_path)
            return True
        except:
            print(file_path)
    else:
        return None


def is_obfuscated_or_encoded(url):
    '''
    Check that the evidence url is not obfuscated or encoded.
    '''
    unquoted_url = url
    try:
        return '%' in unquoted_url or '//' in unquoted_url.split('/')[2]
    except:
        return True


def is_likely_html(url):
    '''
    Check that the evidence url is html
    '''
    # List of common file extensions
    file_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.doc', '.docx', '.ppt', '.pptx', '.xls', 
                       '.xlsx', '.txt', '.zip', '.rar', '.exe', '.svg', '.mp4', '.avi', '.mp3']

    # Extract the extension from the URL
    extension = '.' + url.rsplit('.', 1)[-1].lower()

    # Check if the URL ends with a common file extension
    if extension in file_extensions:
        return False
    else:
        return True
    

def is_fc_organization(url):
    '''
    Check that the evidence url does not come from a FC organization
    Note: the provided list does not include every single existing FC organization. Some FC articles might still pass through this filter.
    '''
    fc_domains = ['https://www.fastcheck.cl','https://pesacheck.org','https://africacheck.org','https://www.snopes.com',
            'https://newsmobile.in', 'https://211check.org', 'factcrescendo.com/', 'https://leadstories.com', 'https://www.sochfactcheck.com', 
            'https://newschecker.in','https://www.altnews.in', 'https://dubawa.org', 'https://factcheck.afp.com', 'factly.in', 
            'https://misbar.com/factcheck/', 'larepublica.pe/verificador/', 'fatabyyano.net/', 'https://www.vishvasnews.com/', "newsmeter.in" , 
            "boomlive", "politifact","youturn.in", "lemonde.fr/les-decodeurs","factuel.afp.com","thequint.com", "logicalindian.com/fact-check/", 
            "timesofindia.com/times-fact-check", "indiatoday.in/fact-check/", "smhoaxslayer.com", "facthunt.in", "aajtak.in/fact-check/",
            "bhaskar.com/no-fake-news", "theprint.in/hoaxposed/", 'firstdraftnews.org',
            "boomlive.in", "correctiv.org", "chequeado.com", "https://maldita.es/",
            "colombiacheck.com/", "mythdetector.com", "bplive.com/fact-check", "logicallyfacts.com", "poligrafo.sapo.pt", 
            "dfrac.org", "congocheck.net", "knack.be/factcheck/", "mythdetector.ge", "youturn.in", "dogrula.org", "verify-sy.com",
            "tjekdet.dk", "factcheckcenter.jp", "factchecker.gr", "factchecklab.org", "pigafirimbi.africauncensored.online",
            "https://www.estadao.com.br/estadao-verifica", "https://www.newscheck.it/", "factindia.in", "indiatimes.com/times-fact-check/",
            "aap.com.au/factcheck/", "ptinews.com/factcheck", "fullfact.org/", "truemedia.org", ".factrakers.org/", "stopfake.org", "digiteye.in",
            "indiacheck.in", "newsmobile.in/nm-fact-checker/", "misbar.com/en/factcheck/", "onlyfact.in", "aosfatos.org", "haqcheck.org",
            "observador.pt/factchecks/", "facta.news", "check4spam.com", "boliviaverifica.bo"
            ]
    for d in fc_domains :
        if d in url:
            return True
    return False


def is_banned(url):
    '''
    Check if the evidence url is in the list of banned urls
    '''
    banned = [
        #Those websites are flagged as potential unsafe or block the webscraping process
        "legalaiddc-prod.qed42.net", "windows8theme.net", "darkroom.baltimoresun.com", "dn.meethk.com", "hotcore.info", "pre-inscription.supmti.ac.ma",
        "hideaways.in", "www.alhurra.com/search?st=articleEx", "anonup.com", "hiliventures", "middleagerealestateagent", "nonchalantslip.fr",
        "corseprestige.com", ".aa.com.tr",  "landing.rassan.ir", "aiohotzgirl.com", "hotzxgirl.com", "//en.rattibha.com/",
        #The content of those social media websites is not scrapable.
        "facebook.com", "twitter.c", "youtube.co", "linkedin.co", "tiktok.c", "quora.c", "gettyimages.", "reddit.", "/x.com/", "soundcloud.com/",
        ".threads.net", "/www.facebook.com", "mstdn.social", "pinterest.com"
        ]
    for b in banned:
        if b in url:
            return True
    return False


def get_filtered_retrieval_results(path, start_idx = 0, num_ev=20):
    '''
    Filter the results of reverse image search.
    Args:
        path (str): path to the file that contains the raw RIS results from Google Reverse Image Search
        start_idx (int): the image to start with (if the download gets interrupted)
        num_ev (int): number of RIS evidence to scrape
    '''
    ris_results = load_json(path)[start_idx:]
    retrieval_results = []
    # Iterate over the URLs and apply the filters
    for i in range(len(ris_results)):
        for u in range(min(len(ris_results[i]['urls']),num_ev)):
            #Loop through all evidence urls, and see if they meet the requirements
            evidence_url = ris_results[i]['urls'][u]
            ris_data = {
                'image_path': ris_results[i]['image_path'], 
                'raw_url': evidence_url,
                'image_urls': ris_results[i]['image_urls'][evidence_url], 
                'is_fc': is_fc_organization(evidence_url),
                'is_banned': is_banned(evidence_url),
                'is_obfuscated': is_obfuscated_or_encoded(evidence_url), 
                'is_html': is_likely_html(evidence_url),
                'is_https': evidence_url.startswith('https')
            }
            # Selection condition
            ris_data['selection'] = ris_data['is_html'] and ris_data['is_https'] and not ris_data['is_obfuscated'] and not ris_data['is_banned'] and not ris_data['is_fc']
            # Append the dictionary to the list if it meets all the criteria
            retrieval_results.append(ris_data)

    # Filter the data based on the selection criteria
    selected_retrieval_results = [d for d in retrieval_results if d['selection']]
    return selected_retrieval_results


def compute_url_distance(url1,url2,threshold):
    distance = lev.distance(url1,url2)
    if distance < threshold:
        return True
    else:
        return False


def find_image_caption(soup, image_url,threshold=25):
    '''
    Retrieve the caption corresponding to an image url by searching the html in BeautifulSoup format.
    '''
    img_tag = None
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or img.get('data-original')
        if src and compute_url_distance(src, image_url, threshold):
            img_tag = img
            break
    if not img_tag:
        return "Image not found"
    figure = img_tag.find_parent('figure')
    if figure:
        figcaption = figure.find('figcaption')
        if figcaption:
            return figcaption.get_text().strip()
    for sibling in img_tag.find_next_siblings(['div', 'p','small']):
        if sibling.get_text().strip():
            return sibling.get_text().strip()
    title = img_tag.get('title')
    if title:
        return title.strip()
    # Strategy 4: Use the alt attribute of the image
    alt_text = img_tag.get('alt')
    if alt_text:
        return alt_text.strip()

    return "Caption not found"


def extract_info_trafilatura(page_url,image_url):
    try:
        headers= {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'} 
        response = requests.get(page_url, headers=headers, timeout=(5,5))
        if response.status_code == 200:
            #Extract content with Trafilatura
            result = bare_extraction(response.text,
                                   include_images=True,
                                   include_tables=False)
            #Remove unnecessary contente
            keys_to_keep = ['title','author','url',
                            'hostname','description','sitename',
                            'date','text','language','image','pagetype']
            result = {key: result[key] for key in keys_to_keep if key in result}
            result['image_url'] = image_url
            # Finding the image caption
            image_caption = []
            soup = bs(response.text, 'html.parser')
            for img in image_url:
                image_caption.append(find_image_caption(soup, img))
            image_caption.append(find_image_caption(soup,result['image']))
            result['image_caption'] = image_caption
            result['url'] = page_url
            # print(result)
            return result
        else:
            return "Failed to retrieve webpage"
    except Exception as e:
        return f"Error occurred: {e}"


def time_difference(date1, date2):
    '''
    Compute whether date1 preceeds date2
    '''
    # Parse the dates
    dt1 = parser.parse(date1)
    dt2 = parser.parse(date2)
    # Make both dates offset-aware, assuming UTC if no timezone is provided
    if dt1.tzinfo is None:
        dt1 = dt1.replace(tzinfo=tzutc())
    if dt2.tzinfo is None:
        dt2 = dt2.replace(tzinfo=tzutc())
    return dt1 < dt2


def merge_data(evidence, evidence_metadata,dataset, apply_filtering=False):
    '''
    Merge all evidence by dropping duplicates and applying 2 filters:
    1) The evidence is not the original FC article itself
    2) The evidence has been published before the FC article
    '''
    evidence_df = pd.DataFrame(evidence)
    # print(evidence_df.columns)
    evidence_metadata_df = pd.DataFrame(evidence_metadata)
    # print(evidence_metadata_df.columns)
    dataset_df = pd.DataFrame(dataset)
    merged_data = pd.merge(evidence_df, evidence_metadata_df.drop_duplicates(subset='raw_url')[['image_path','raw_url']].rename(columns={'raw_url':'url'}), on='url',how='inner')
    merged_data = pd.merge(merged_data.rename(columns={'url':'evidence_url'}), 
                           dataset_df[['fc_org','image_path','fc_pub_date']].rename(columns={'fc_pub_date': 'date_filter'}), 
                           on='image_path',how='inner')
    merged_data  = merged_data.dropna(subset='evidence_url')
    #Apply optional filtering steps
    if apply_filtering:
        #Verify that the evidence is not the FC article itself.
        fc_mask = merged_data.apply(lambda row : False if row['fc_org'] in row['evidence_url'] or row['fc_org'] in ''.join(row['image_url']) else True, axis=1)
        merged_data = merged_data[fc_mask]
        #Remove evidence that have been published after the FC article or have no publication date
        merged_data = merged_data[~merged_data['date'].isnull()]
        time_mask = merged_data.apply(lambda row : time_difference(row['date'],row['date_filter']),axis=1)
        merged_data = merged_data[time_mask]   
    merged_data = merged_data[['image_path','fc_org','evidence_url','title','author','hostname',
                           'description', 'text','sitename','date','image','image_url','image_caption']]
    merged_data = merged_data.drop_duplicates(subset=['evidence_url','image_path'])
    return merged_data
