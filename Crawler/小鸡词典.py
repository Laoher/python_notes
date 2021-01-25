import requests
import re
from bs4 import BeautifulSoup


def most_simple_crawl():
    # easiet crawler
    content = requests.get('https://jikipedia.com/definition/962417117').content
    soup = BeautifulSoup(content, 'html.parser')
    for div in soup.find_all('div', {'class': 'content'}):
        print(div.text.strip())


if __name__ == '__main__':
    most_simple_crawl()
