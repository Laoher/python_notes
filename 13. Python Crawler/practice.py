from requests_html import HTMLSession
from bs4 import BeautifulSoup
import requests

# session = HTMLSession()
#
# r = session.get('https://python.org/')
# print(r.html.absolute_links)
# about = r.html.find('#about',first=True)
# print(about.text)
response = requests.get('http://www.4j4j.cn/')
soup = BeautifulSoup(response.text)
print()
for link in soup.find_all('p'):
    for a in link.find_all("a"):
        print(a)
        print(a.get('href'))
        print(a.string)


