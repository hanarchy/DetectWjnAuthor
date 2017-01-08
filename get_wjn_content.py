# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
import pickle
import os
import time
import re
import sys


WJN_BASE_URL = 'http://wjn.jp'
WJN_URL = 'http://wjn.jp/article/category/7/'
index_list_pkl = 'wjn_cat7_index_link_list.pkl'
sys.setrecursionlimit(1000000)


class WjnContents():
    def __init__(self):
        ksk_url_list = get_ksk_url_list()
        ksk_contents = get_ksk_contents(ksk_url_list)
        self.contents, self.labels = detect_author(ksk_contents)


def get_ksk_url_list():
    # wjnのサイトから奈倉氏、柏木氏が書く記事である、
    # 官能小説作家書き下ろし[実録]がタイトルにつく記事のURLをlist化
    # TODO:取得データ数を増やす
    # wjnのサイトからは1000件までしか記事のURLが記録されてないが、
    # 過去の記事自体は残っているので、そのURLをどうにかして取得する
    # googleでsite:wjn.jp/article/detail/ "官能小説作家書き下ろし"
    # と検索したら2000件くらい出てきたのでそれを使う感じで

    wjn_cat7_index_link_list = []
    if os.path.exists(index_list_pkl):
        with open(index_list_pkl, 'rb') as f:
            wjn_cat7_index_link_list = pickle.load(f)
    else:
        for i in range(1, 101):
            if i == 1:
                url = WJN_URL
            else:
                url = WJN_URL + str(i)
            print(url)

            html = requests.get(url, headers={'User-Agent': 'test'})
            soup = BeautifulSoup(html.content, "html5lib",
                                 from_encoding='Shift-JIS')
            wjn_cat7_index_link_list.extend(soup.find_all('a',
                                                          {"class": "title"}))
            time.sleep(1)
        with open(index_list_pkl, 'wb') as f:
            pickle.dump(wjn_cat7_index_link_list, f)
    ksk_url_list = [WJN_BASE_URL+a.get('href') for a in
                    wjn_cat7_index_link_list
                    if re.search(r"官能小説作家書き下ろし", a.text)]
    return ksk_url_list


def get_ksk_contents(ksk_url_list, single=False):
    # 記事から本文を抜き出して、pickle化
    if single:
        contents = []
        for url in ksk_url_list:
            html = requests.get(url, headers={'User-Agent': 'test'})
            soup = BeautifulSoup(html.text,
                                 "html5lib")
            content = soup.find("p", {"class": "desc"})
            contents.append(content)
            time.sleep(1)
    else:
        contents = []
        if os.path.exists('wjn_content.pkl'):
            with open('wjn_content.pkl', 'rb') as f:
                contents = pickle.load(f)
        else:
            for url in ksk_url_list:
                html = requests.get(url, headers={'User-Agent': 'test'})
                soup = BeautifulSoup(html.text,
                                     "html5lib", from_encoding="Shift-JIS")
                content = soup.find("p", {"class": "desc"})
                # print(content.text)
                contents.append(content)
                time.sleep(1)
            with open('wjn_content.pkl', 'wb') as f:
                pickle.dump(contents, f)
    return contents


def detect_author(ksk_contents):

    author_removed_contents = []
    author_labels = []
    for content in ksk_contents:
        # print(content.text)
        # print(str(content))
        if re.search(r"柏木春人", content.text):
            author_removed_content = re.sub(r"柏木春人", "", content.text)
            author_removed_contents.append(author_removed_content)
            author_labels.append(1)
        elif re.search(r"奈倉清孝", content.text):
            author_removed_content = re.sub(r"奈倉清孝", "", content.text)
            author_removed_contents.append(author_removed_content)
            author_labels.append(0)

    assert len(author_removed_contents) == len(author_labels)
    return author_removed_contents, author_labels
