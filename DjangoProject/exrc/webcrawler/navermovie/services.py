import csv
import urllib
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

from exrc.webcrawler.music.models import ScrapVO
import os.path

class ScrapService(ScrapVO):
    def __init__(self):
        global driverpath, naver_url, savepath, encoding
        driverpath = r'C:\Users\AIA\MsaProject\DjangoProject\exrc\webcrawler\chromedriver.exe'
        savepath = r'C:\Users\AIA\MsaProject\DjangoProject\exrc\webcrawler\navermovie\movie.csv'
        naver_url = 'https://movie.naver.com/movie/sdb/rank/rmovie.naver'
        encoding = 'UTF-8'

    def bugs_music(self,arg): # BeautifulSoup 기본 크롤링
        soup = BeautifulSoup(urlopen(arg.domain + arg.query_string), 'lxml')
        title = {"class": arg.class_names[0]}
        artist = {"class": arg.class_names[1]}
        titles = soup.find_all(name=arg.tag_name, attrs=title)
        titles = [i.find('a').text for i in titles]
        artists = soup.find_all(name=arg.tag_name, attrs=artist)
        artists = [i.find('a').text for i in artists]
        [print(f"{i}위 {j} : {k}")  # 디버깅
         for i, j, k in zip(range(1, len(titles)), titles, artists)]
        diction = {}  # dict 로 변환
        for i, j in enumerate(titles):
            diction[j] = artists[i]
        arg.diction = diction
        arg.dict_to_dataframe()
        arg.dataframe_to_csv()  # csv파일로 저장

    def melon_music(self,arg): # BeautifulSoup 기본 크롤링
        soup = BeautifulSoup(
            urlopen(urllib.request.Request(arg.domain + arg.query_string, headers={'User-Agent': "Mozilla/5.0"})),
            "lxml")
        title = {"class": arg.class_names[0]}
        artist = {"class": arg.class_names[1]}
        titles = soup.find_all(name=arg.tag_name, attrs=title)
        titles = [i.find('a').text for i in titles]
        artists = soup.find_all(name=arg.tag_name, attrs=artist)
        artists = [i.find('a').text for i in artists]
        [print(f"{i}위 {j} : {k}")  # 디버깅
         for i, j, k in zip(range(1, len(titles)), titles, artists)]
        diction = {}  # dict 로 변환
        for i, j in enumerate(titles):
            diction[j] = artists[i]
        arg.diction = diction
        arg.dict_to_dataframe()
        arg.dataframe_to_csv()  # csv파일로 저장

    def naver_movie_review(self):
        if os.path.isfile(savepath) == True:
            df = pd.read_csv(savepath)
            # df = df.transpose()  # 행 열 전환
            # df = df.reset_index()
            # df.index=df.index+1

            result = [{'rank' : f'{i+1}위', 'title': f'{j}'} for i,j in enumerate(df)] #협업시(리액트로 던질때) 리스트로 던져줘야
            print(result)
            return result
        elif os.path.isfile(savepath) == False:
            driver = webdriver.Chrome(driverpath)
            driver.get(naver_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            all_divs = soup.find_all('div', attrs={'class', 'tit3'})
            products = [[div.a.string for div in all_divs]]
            with open(savepath, 'w', newline='', encoding=encoding) as f:
                wr = csv.writer(f)
                wr.writerows(products)
            driver.close()
            return '크롤링 완료'
        else:
            print("Nothing")



    def show_csv(self):
        f = open(savepath, 'r', encoding=encoding)
        rdr = csv.reader(f)
        for line in rdr:
            return line
        f.close()



stroke_menu = ["Exit",  # 0
               "naver_movie_review",# 1
               "show_csv" #2
               ]
stroke_lambda = {
    "1": lambda x: x.naver_movie_review(),
    "2": lambda x: x.show_csv(),

}
if __name__ == '__main__':
    stroke = ScrapService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(stroke_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                stroke_lambda[menu](stroke)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")