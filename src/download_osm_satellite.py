from staticmap import StaticMap,  IconMarker
import argparse, time
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from _utils import *



def get_static_map(output_file, lat, lon, width=800, height=600, zoom=15):
    m = StaticMap(width, height, url_template='https://a.tile.openstreetmap.org/{z}/{x}/{y}.png')
    transparent_icon = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    buffer = BytesIO()
    transparent_icon.save(buffer, format='PNG')
    buffer.seek(0)
    dummy_marker = IconMarker((lon, lat), buffer, 0, 0)
    m.add_marker(dummy_marker)
    
    image = m.render(zoom=zoom)
    image.save(f"data/staticmap/osm/{output_file}")


def get_satellite_map(api_key, output_file, lat, lon, width=800, height=600, zoom=15):
    """
    Generates a static satellite map image centered on the given latitude and longitude.
    """
    esri_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    if api_key!='':
        esri_url += f'?token={api_key}'

    m = StaticMap(width, height, url_template=esri_url)

    # Create a 1x1 transparent image and write it to a BytesIO buffer.
    transparent_icon = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    buffer = BytesIO()
    transparent_icon.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Add a dummy marker using the BytesIO object.
    dummy_marker = IconMarker((lon, lat), buffer, 0, 0)
    m.add_marker(dummy_marker)
    
    image = m.render(zoom=zoom)
    image.save(f"data/staticmap/satellite/{output_file}")





if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Download images based on provided URLs.')
    parser.add_argument('--api_key', type=str, default=" ",  #Provide here your ESRI API key to download satellite images
                        help='Key to access the ESRI World Imagery API.')
    parser.add_argument('--zoom', type=int, default=15, 
                        help='Zoom level for satellite and map.')
    parser.add_argument('--sleep', type=int, default=2,
                        help='Waiting time in seconds between scraping two images.')
    
    args = parser.parse_args()

    data = load_json('data/M4FC.json')
    geoloc_coordinates = set([d['coordinates'] for d in data if d['coordinates']!= 'not enough information'] + [co for d in data for co in d['negative_coordinates'] if d['negative_coordinates']!='not enough information'])

    os.makedirs('data/staticmap/', exist_ok=True)
    os.makedirs('data/staticmap/osm/', exist_ok=True)
    os.makedirs('data/staticmap/satellite/', exist_ok=True)
    for co in tqdm(geoloc_coordinates):
        lat, lon = float(co.split('/')[0]), float(co.split('/')[1])
        output_file = '-'.join(co.split('/')) + '-' + str(args.zoom)  + '.png'
        if output_file not in os.listdir('data/staticmap/osm'):
            get_static_map(lat, lon, zoom=args.zoom, output_file=output_file)
            get_satellite_map(args.api_key, lat, lon, zoom= args.zoom, output_file=output_file)
            time.sleep(5)