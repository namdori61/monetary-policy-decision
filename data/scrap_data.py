from typing import List
import os
from random import uniform
from time import sleep

import requests
from bs4 import BeautifulSoup
from absl import app, flags, logging
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', default=None,
                    help='Path to save dataset')
flags.DEFINE_integer('page_num', default=25,
                     help='Page numbers to scrap')


def extract_detail_page_url(page_num: int = 25) -> List[str]:
    detail_page_url_list = []
    for i in tqdm(range(1, page_num+1), desc='scrapping'):
        sleep(uniform(1, 5))
        url = f'https://www.bok.or.kr/portal/bbs/P0000093/list.do?menuNo=200789&searchWrd=%ED%86%B5%ED%99%94%EC%A0%95%EC%B1%85%EB%B0%A9%ED%96%A5&searchCnd=1&sdate=&edate=&pageIndex={i}'
        list_page = requests.get(url)
        if list_page.ok:
            soup = BeautifulSoup(list_page.content, 'html.parser')
            detail_pages = soup.find_all('span', class_='col m10 s10 x9 ctBx')

            for page in detail_pages:
                if page:
                    detail_page_url_list.append('https://www.bok.or.kr' + page.find('a')['href'])
                else:
                    raise ValueError('BOK detail page url pattern changed in list page.')
        else:
            logging.info(f'The {i}th list page is not available: status code {list_page.status_code}.')

    return detail_page_url_list


def download_data_from_urls(url_list: List[str] = None,
                            save_dir: str = None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for url in tqdm(url_list, desc='downloading'):
        sleep(uniform(1, 5))
        detail_page = requests.get(url)
        if detail_page.ok:
            soup = BeautifulSoup(detail_page.content, 'html.parser')
            file_class = soup.find('div', class_='addfile')
            file_tag_list = file_class.find_all('a')
            if any('hwp' in s.text for s in file_tag_list):
                for file_url in file_tag_list:
                    if 'hwp' in file_url.text.lower():
                        file_download_url = 'https://www.bok.or.kr' + file_url['href']
                        file_date = soup.find('span', class_='date').text[-10:].replace('.', '-')
                        file_name = 'MPD' + file_date + '.hwp'
                        file_path = os.path.join(save_dir, file_name)
                        file = requests.get(file_download_url)
                        if file.ok:
                            logging.info(f'saving to {os.path.abspath(file_path)}')
                            with open(file_path, 'wb') as f:
                                f.write(file.content)
            else:
                logging.info('No hwp file.')
        else:
            raise ValueError(f'BOK detail page is not available: status code {detail_page.status_code}')


def main(argv):
    logging.info(f'Scrapping {FLAGS.page_num} list pages.')
    detail_page_url_list = extract_detail_page_url(page_num=FLAGS.page_num)

    logging.info(f'Downloding {len(detail_page_url_list)} files from url to {FLAGS.save_dir}')
    download_data_from_urls(url_list=detail_page_url_list,
                            save_dir=FLAGS.save_dir)


if __name__ == '__main__':
    flags.mark_flags_as_required(['save_dir'])
    app.run(main)