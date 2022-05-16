import os
import requests
from bs4 import BeautifulSoup
import urllib
import numpy as np
import json
from tqdm import tqdm

# make opener
def make_scrape_profile():
    # set up a profile to access website
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


# -------------------------
# ARBOR DAY TREE SCRAPING
def get_arbor_tree_imageurls(url, tree):
    # load page
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    image_tags = soup.find_all('img')
    # get all image links
    links = []
    for image_tag in image_tags:
        links.append(image_tag['src'])
    # subset to tree image links
    tree_links = [link for link in links if any(word in link.lower() for word in tree.split('-'))]
    return tree_links

def download_arbor_tree_images(path, trees):
    for t in trees:
       
        # set up tree and url
        th = t.replace(' ', '-')
        tn = t.replace(' ', '')
        url = f'https://shop.arborday.org/{th}'

        # get links to tree images
        tree_links = get_arbor_tree_imageurls(url, th)
        print(f'pulling {len(tree_links)} {t}s from {url}')

        # ARBOR DAY SPECIFIC: for each image, substitute thumbnail for large image
        tree_links = [link.replace('105.jp','510.jp') for link in tree_links]

        # loop through images and download them
        image_path = path / tn
        for i, l in enumerate(tree_links):
            with urllib.request.urlopen(l, timeout=5) as urlopener:
                raw_img = urlopener.read()

            filename = f'arborday-{tn}-{i}.jpg'
            with open(os.path.join(image_path, filename), 'wb') as f:
                f.write(raw_img)
                f.close()

# -------------------------
# HARVARD ARBORETUM TREE SCRAPING

def get_harvard_tree_imageurls(search_url, image_base_url):
    page = requests.get(search_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    page_tags = soup.find_all('img')
    
    # get all image links
    search_links = []
    for page_tags in page_tags:
        search_links.append(page_tags['src'])

    # subset to links from the plant library
    tree_search_links = [link for link in search_links if '/plant/img/' in link]
    
    # get large images
    tag_start = '/img/aaimg/'
    tag_end = '.mid_200'
    tree_links = []
    for tl in tree_search_links:
        if tl.endswith('lg.jpg'):
            tree_links += [tl]
        else:
            image_page_label = tl[tl.find(tag_start)+len(tag_start):tl.find(tag_end)]
            image_page_url = image_base_url + image_page_label
            page = requests.get(image_page_url)
            soup = BeautifulSoup(page.content, 'html.parser')
            image_tags = soup.find_all('img')
            for image_tags in image_tags:
                tree_links.append(image_tags['src'])
    
    return tree_links

def download_harvard_tree_images(path, trees, search_trees):
    for t, ts in zip(trees, search_trees):
        tsp = ts.replace(' ', '+')
        tn = t.replace(' ', '')
        # get list of tree URLs
        search_url = f'http://arboretum.harvard.edu/plants/image-search/?keyword={tsp}&image_per_page=1000'
        image_base_url = f'https://arboretum.harvard.edu/plants/image-search/?keyword={tsp}&search_type=indiv_img&image_key='
        tree_links = get_harvard_tree_imageurls(search_url, image_base_url)
        print(f'pulling {len(tree_links)} {t}s from {search_url}')

        # loop through images and download them
        image_path = path / tn
        for i, l in enumerate(tree_links):
            with urllib.request.urlopen(l, timeout=5) as urlopener:
                raw_img = urlopener.read()

            filename = f'harvard-{tn}-{i}.jpg'
            with open(os.path.join(image_path, filename), 'wb') as f:
                f.write(raw_img)
                f.close()

# -------------------------
# BING IMAGE SCRAPING

# https://gist.github.com/stephenhouser/c5e2b921c3770ed47eb3b75efbc94799
def get_soup(url,header):
    #return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),
    # 'html.parser')
    return BeautifulSoup(urllib.request.urlopen(
        urllib.request.Request(url,headers=header)),
        'html.parser')

def get_bing_imageurls(query, num_pages=1, start_page=0, per_page=75):
    '''query : search terms, as they would appear in Bing'''
    query= query.split()
    query='+'.join(query)
    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    
    ActualImages=[]# contains the link for Large original images, type of  image
    
    url = f'http://www.bing.com/images/search?q={query}qft=+filterui:imagesize-large&form=IRFLTR'
    url_II = f'http://www.bing.com/images/search?q={query}&first=1000' #qft=+filterui:imagesize-large&form=IRFLTR'
    for n in range(1, num_pages+1):
        url_II = f'https://www.bing.com/images/async?q={query}&first={per_page*n+start_page}&count={per_page}'
        soup = get_soup(url_II, header)

        for a in soup.find_all("a",{"class":"iusc"}):
            m = json.loads(a['m'])
            murl = m['murl']
            jpg_loc = murl.lower().find('.jpg')
            if jpg_loc > 0:
                murl = murl[0:jpg_loc+4]

            image_name = urllib.parse.urlsplit(murl).path.split("/")[-1]
            image_desc = m['desc']

            ActualImages.append((image_name, murl, image_desc))

    print(f'pulled {len(ActualImages)} image urls\n---- Downloading ----')
    
    return ActualImages



def save_bing_images(label, image_urls, path):

    image_path = path / label
    metadata_path = path / 'metadata'

    if not os.path.exists(image_path):
        os.mkdir(image_path)
    # if not os.path.exists(metadata_path):
    #     os.mkdir(metadata_path)

    ##print images
    image_descs = []
    load_exceptions = {}
    load_true = 0
    for i, (image_name, murl, image_desc) in enumerate(tqdm(image_urls)):
        
        try:
            with urllib.request.urlopen(murl, timeout=5) as urlopener:
                raw_img = urlopener.read()
            
            filename = 'bing-' + image_name[:60].lower().replace('.jpg', '').replace('.jpeg', '') + '.jpg'
            with open(os.path.join(image_path, filename), 'wb') as f:
                f.write(raw_img)
                f.close()

            image_descs.append((image_name, image_desc))
            load_true += 1
        except Exception as e:
            try:
                load_exceptions[e] += 1
            except:
                load_exceptions[e] = 1
            continue
            
    
    print(f'Successfully loaded {load_true} images')
#     print(f'Excpetions:', load_exceptions)
    # np.save(metadata_path/f'bing-{label}', image_desc)


def download_bing_images(path, search_term_array, num_pages, photos_per_page):

    for s in search_term_array:
        print(f'==== Scraping {s} ====')
        image_urls = get_bing_imageurls(s, num_pages=num_pages, per_page=photos_per_page)
        save_bing_images(label=s.replace(' ', ''), image_urls=image_urls, path=path)