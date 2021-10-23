from dotenv import load_dotenv
import os
from serpapi import GoogleSearch
import pandas as pd
from glob import glob
import ast
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
import sys

tqdm.pandas()

load_dotenv()
SERP_API_KEY = os.getenv('SERP_API_KEY')
SCRAPERAPI_API_KEY = os.getenv('SCRAPERAPI_API_KEY')

# signup for serp_api for api key to search google scholar.
# signup for scraperapi to deal with captcha issue with certain publishers.

params = {
    'engine': 'google_scholar',
    'api_key': SERP_API_KEY
}
file_name = 'query_results'
folder_name='data'

def process_res(res):
    return pd.DataFrame(res['organic_results'])


def query_google(query):
    params['q'] = query
    params['num'] = 4
    return GoogleSearch(params).get_json()


def save_query_results(df, query):
    os.makedirs(folder_name, exist_ok=True)
    df.to_csv(os.path.join(folder_name,f'{file_name}_{query.replace(" ", "_")}.csv'), index=False)


def query_google_scholar(query):
    res = query_google(query)
    df = process_res(res)
    save_query_results(df, query)


def add_query_value(df, file_name):
    df['query'] = file_name.split('.')[0]
    return df


def combine_query_results():
    files = glob(os.path.join(folder_name,'*.csv'))
    df = pd.concat([add_query_value(pd.read_csv(f), f)
                    for f in files if f != 'query_results.csv'])
    df = df.drop(['result_id', 'position', 'inline_links', 'type'], axis=1)
    return df


def get_source_details(df, value_type, column_name):
    df[column_name] = df.apply(lambda x: ast.literal_eval(x['resources'])[
        0][value_type] if pd.notnull(x['resources']) else '', axis=1)
    return df


def process_combined_result(df):
    df = get_source_details(df, 'title', 'source_name')
    df.drop(['publication_info', 'resources'], axis=1, inplace=True)
    df = df[~df['link'].str.contains('book')]
    df=df.drop_duplicates()
    return df


def get_tag_property(source):
    tag = ['div']
    sel_value = None
    use_scraper_api = False
    is_ieee = False
    if source == 'sciencedirect.com':
        sel_value = [{'id':'abstracts'}]
    if source == 'link.springer.com':
        sel_value = [{'id':'Abs1-section'}]
    elif source == 'journals.lww.com':
        sel_value = [{'id': 'panel1'}]
    elif source == 'frontiersin.org':
        sel_value = [{'class':'JournalAbstract'}]
    elif source == 'journals.plos.org':
        sel_value = [{'class':'abstract-content'}]
    elif source == 'biomedical-engineering-online.biomedcentral.com':
        sel_value = [{'id':'Abs1-content'}]
    elif source == 'academic.oup.com':
        tag = ['section']
        sel_value = [{'class':"abstract"}]
        use_scraper_api = True
    elif source == 'onlinelibrary.wiley.com':
        sel_value = [{'class':"article-section__content en main"}]
        use_scraper_api = True
    elif source == 'jneurosci.org':
        sel_value = [{'id':'abstract-1'}]
    elif source == 'ajp.psychiatryonline.org':
        sel_value = [{'class': "abstractSection abstractInFull"}]
    elif source == 'ieeexplore.ieee.org':
        sel_value = [{'class':'abstract-text'}]
        is_ieee = True
    elif source == 'cambridge.org':
        sel_value = [{'class': 'abstract-text-container'}]
    elif source == 'nature.com':
        tag=['div','div']
        sel_value = [{'id': 'Abs1-content'}, {'id': 'Abs2-content'}]
    elif source == 'pnas.org':
        sel_value = [{'id': 'abstract-1'}]
    if sel_value:
        return tag, sel_value, use_scraper_api, is_ieee
    return sel_value



def get_payload(link):
    return {'api_key': SCRAPERAPI_API_KEY, 'url': link}


def scrap_abstract(link, source):
    """
    Scrap the web page for the abstract
    """
    if link == '':
        return link
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36"
    }
    props = get_tag_property(source)
    if props:
        if props[2]:
            page = requests.get('http://api.scraperapi.com',
                                params=get_payload(link))
        else:
            page = requests.get(link, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        if props[3]:
            abstract = soup.find_all('meta')
            if abstract:
                return abstract[5]['content']
        else:
            abstract=None
            ind=0
            while not abstract and ind<len(props[1]):
                abstract = soup.find(props[0][ind], props[1][ind])
                ind+=1
            if abstract:
                return abstract.text.strip()
    return np.nan


def print_final_report(df):
    print(f'Results found: {df.shape[0]}')
    print(f'Abstracts not found: {df.isna().sum()["abstract"]}')
    print(f'Resultant file name: {file_name}')


if __name__ == '__main__':
    print('Querying results')
    for query in sys.argv[1:]:
        query_google_scholar(query)
    print('Results gathered, combining query results.')
    combined_result = combine_query_results()
    combined_result = process_combined_result(combined_result)
    combined_result['abstract']=np.NaN
    print('Extracting abstracts')
    combined_result['abstract'] = combined_result.progress_apply(lambda x: scrap_abstract(
        x['link'], x['source_name']) if pd.isna(x['abstract']) else x['abstract'], axis=1)
    print('Saving Final Result')
    combined_result.to_csv(os.path.join(folder_name,f'{file_name}.csv'),index=False)
    print_final_report(combined_result)
