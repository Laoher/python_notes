#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import requests
import os
from bs4 import BeautifulSoup

base_url = 'https://www.lockhatters.co.uk/women.html'

# 获取每个帽子的url
def get_img():
    response = requests.get(base_url)
    response.encoding = 'utf-8'
    html = BeautifulSoup(response.text, 'html.parser')
    i = 1
    save_path = r'D:\labeling & extracting & downloading tools\crawler\pic\hat1'
    os.chdir(save_path)
    data = html.find('ul', {'class': 'products-grid'})
    for hat in data.find_all('img'):
        print(hat)
        hat_url = hat.get('data-small_image')
        res = requests.get(hat_url)
        res.encoding = 'utf-8'
        if res.status_code == 200:
            with open('pic_%d.jpg' % i, 'wb') as fp:
                fp.write(res.content)
                i += 1


if __name__ == '__main__':
    get_img()
