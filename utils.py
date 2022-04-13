import numpy as np
import pandas as pd

import requests
import re
from bs4 import BeautifulSoup

import logging


def downcast_dtypes(df):
    # 데이터 용량 줄이는 함수 정의 (64비트 -> 32비트)
    float_cols = [c for c in df if df[c].dtype in ["float64"]]
    int_cols = [c for c in df if df[c].dtype in ["int64"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)
    return df


def hoilday_info(key, is_crawling=True):
    # 공공데이터포털 특일 정보에서 공휴일 데이터를 가져와서 데이터프레임으로 정리
    holidays = {'date': [], 'name': []}

    if is_crawling:
        for target_year in ['2018', '2019', '2020']:
            for target_month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
                print(target_year, target_month)

                url = f'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/' \
                      f'getRestDeInfo?serviceKey={key}&solYear={target_year}&solMonth={target_month}'
                resp = requests.get(url)
                soup = BeautifulSoup(resp.text, 'html.parser')
                items = soup.findAll('item')
                print(len(items))
                if len(items) > 0:
                    for item in items:
                        holidays['date'].append(item.locdate.text)
                        holidays['name'].append(item.datename.text)

        holiday_df = pd.DataFrame(holidays)
    else:
        holiday_df = pd.read_csv('holiday_df.csv', encoding='euc-kr')
        holiday_df['date'] = holiday_df['date'].astype(str)

    return holiday_df


def tourist_info(is_crawling=True):
    if is_crawling:
        # 제주특별자치도 정보공개게시판의 관광객 입도 현황(90) 자료를 크롤링
        seq_list = [1100943, 1100944, 1103894, 1107414, 1111003, 1114289, 1121668, 1121669, 1163037, 1163039, 1163567,
                    1166992, 1170414, 1174074, 1178448, 1184531, 1187140, 1192974, 1198268, 1201201, 1222916, 1226799,
                    1227644, 1234836, 1238511, 1242775, 1242777]
        dates = []
        people = []

        for page in [1, 2]:
            for seq in seq_list:
                url = "https://www.jeju.go.kr/open/open/iopenboard.htm?category=1035&page=%s&act=view&seq=%s" % (page, seq)
                try:
                    resp = requests.get(url)
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    table = soup.find('td', {'class': 'article-contents left'})
                    p_items = table.select('p')
                    if len(p_items) > 5:
                        y = re.compile('([0-9]+)년').findall(table.text)[0]
                        m = re.compile('([0-9]+)월').findall(table.text)[0]
                        if len(m) == 1:
                            m = '0' + m
                        date = y + '/' + m
                        p = re.compile('([0-9]+.[0-9]+,[0-9]+)').findall(table.text)[1].replace(',', '')
                    elif len(p_items) == 0:
                        y = re.compile('([0-9]+)년').findall(table.text)[0]
                        m = re.compile('([0-9]+)월').findall(table.text)[0]
                        if len(m) == 1:
                            m = '0' + m
                        date = y + '/' + m
                        p = re.compile('([0-9]+.[0-9]+,[0-9]+)').findall(table.text)[1].replace(',', '')
                    else:
                        y = re.compile('([0-9]+)년').findall(table.text)[0]
                        m = re.compile('([0-9]+)월').findall(table.text)[0]
                        if len(m) == 1:
                            m = '0' + m
                        date = y + '/' + m
                        p = re.compile('([0-9]+.[0-9]+,[0-9]+)').findall(table.text)[0].replace(',', '')

                    if date in dates:
                        pass
                    else:
                        dates.append(date)
                        people.append(p)
                        print(date, p, '명')

                except Exception as e:
                    logging.exception(e)

        tourist = pd.DataFrame({'time': dates, 'tourist': people})
        tourist['tourist'] = tourist['tourist'].astype(int) / 1000000
        tourist[['year', 'month']] = tourist['time'].str.split('/', expand=True)
        tourist[['year', 'month']] = tourist[['year', 'month']].astype(int)

        # 5월, 6월 데이터를 추정 : 전년도 대비 -40% 적용 (코로나19로 인한 입도객 감소 트렌드 반영)
        tourist.loc[27] = ['2020/05', tourist.loc[tourist.time == '2019/05', 'tourist'].values[0] * 0.6, 2020, 5]
        tourist.loc[28] = ['2020/06', tourist.loc[tourist.time == '2019/06', 'tourist'].values[0] * 0.6, 2020, 6]

        tourist = tourist.drop(['time'], axis=1)
    else:
        tourist = pd.read_csv('tourist_data.csv')

    return tourist


def calc_rmsse(y_true, y_pred, y_hist, axis=None, weight=None):
    """
    y_true: 실제 값
    y_pred: 예측 값
    y_hist: 과거 값 (public LB는 v1 기간으로 계산, private LB는 v2 기간으로 계산)
    """

    # axis = 0
    # weight = [0.1, 0.1, 0.2, 0.6] (smp_max, smp_min, smp_mean, supply에 대한 가중치)

    y_true, y_pred, y_hist = np.array(y_true), np.array(y_pred), np.array(y_hist)
    h, n = len(y_true), len(y_hist)

    numerator = np.sum((y_true - y_pred) ** 2, axis=axis)
    denominator = 1 / (n - 1) * np.sum((y_hist[1:] - y_hist[:-1]) ** 2, axis=axis)

    msse = 1 / h * numerator / denominator
    rmsse = msse ** 0.5
    score = rmsse

    if weight is not None:
        score = rmsse.dot(weight)

    return score