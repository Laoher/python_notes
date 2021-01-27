#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import requests
import os
from bs4 import BeautifulSoup

base_url = 'http://www.4j4j.cn'
index_url = 'http://www.4j4j.cn/beauty/index.html'

# 获取每个美女详情页的url
def get_url_list():
    response = requests.get(base_url)
    response.encoding = 'utf-8'
    html = BeautifulSoup(response.text, 'html.parser')
    result =[]
    for link in html.find_all('p'):
        for a in link.find_all("a"):
            result.append([(a.get('href'), a.string)])
    print(result)
    return result

# 下载图片保存到本地
def get_img(beauty_url, title):
    save_path = r'D:\labeling & extracting & downloading tools\crawler\pic\beauty' + "\\"+title
    os.mkdir(save_path)
    os.chdir(save_path)
    print(os.getcwd())
    response = requests.get(beauty_url)
    response.encoding = 'utf-8'
    html = BeautifulSoup(response.text, 'html.parser')
    data = html.find('div', {'class': 'beauty_details_imgs_box'})
    girls = data.find_all('img')
    i = 1
    for girl in girls:
        girl_url = girl['src']
        res = requests.get(girl_url)
        res.encoding = 'utf-8'
        if res.status_code == 200:
            with open('pic_%d.jpg' % i, 'wb') as fp:
                fp.write(res.content)
                i += 1


def get_page():
    url_list = get_url_list()
    print(url_list)
    for url in url_list:
        print(url)
        beauty_url = base_url+url[0][0]
        title = url[0][1]
        print(beauty_url)
        print(title)
        get_img(beauty_url=beauty_url, title=title)

if __name__ == '__main__':
    get_page()
