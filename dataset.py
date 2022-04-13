import pandas as pd

import missingno as msno
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import plotly.express as px

import math
import utils
import numpy as np
import os
from datetime import datetime
from datetime import timedelta


AUTH_KEY = "Get personal authorization key"


class DataSet:
    def __init__(self, user_inp):
        self.target = None
        self.smp = None
        self.weather = None
        self.lookup = None
        self.energy_cost = None

        self.user_inp = user_inp

        self.base_dir = user_inp.drive_path
        self.data_dir = os.path.join(self.base_dir, 'data')

    def read_data(self):
        os.chdir(self.data_dir)
        # 제공 csv 데이터 파일을 읽어서 데이터프레임으로 변환
        start, end = self.user_inp.data_start, self.user_inp.data_end
        tmp_target = pd.read_csv('target.csv')
        target_mask = (tmp_target['date'] >= start) & (tmp_target['date'] <= end)
        self.target = tmp_target.loc[target_mask, :]

        self.smp = pd.read_csv('hourly_smp.csv')
        self.weather = pd.read_csv('weather.csv')
        self.lookup = pd.read_csv('lookupTable_area.csv')
        self.energy_cost = pd.read_csv('energy_cost.csv', encoding='euc-kr')

        # self.submission = pd.read_csv('sample_submission.csv')
        # print("Loading the raw data from source...")
        # print("target: ", self.target.shape)
        # print("smp: ", self.smp.shape)
        # print("lookup: ", self.lookup.shape)
        # print("weather: ", self.weather.shape)
        # print("sub: ", self.submission.shape)

    def show_asos_area(self):
        # ASOS 관측소
        print("ASOS 관측소")
        asos_area = self.weather.loc[self.weather['station'] == 'ASOS', 'area'].unique()
        print(asos_area)

        for area in asos_area:
            print(area)
            print((self.lookup.loc[self.lookup['area'] == area, 'name']))

    def show_missing_value(self, plot=False):
        # 결측값 구성비
        print("각 관측소의 결측 비율, 최소 결측 지점: 1.0 기준")
        print(self.weather['area'].value_counts(dropna=False) / self.weather['area'].value_counts(dropna=False).max())

        if plot:
            msno.matrix(self.weather, figsize=[12, 7], fontsize=13)
            plt.show()

            msno.bar(self.weather, figsize=[12, 7], fontsize=8)
            plt.show()

    def show_avg_correlation(self, plot=False):
        # 평균 상관관계 확인
        self.weather['date'] = self.weather['datetime'].apply(lambda x: x[:10])
        weather_mean = self.weather.groupby('date').mean().reset_index(drop=True)

        # weather_mean = weather_mean[["temp", "ws", "wd"]]

        merged_table = pd.concat([self.target, weather_mean], axis=1)
        corr_all = merged_table.corr()
        corr_selected = corr_all.loc['smp_max':'supply', 'area':]
        avg_corr = corr_selected.mean()
        sorted_corr = avg_corr.sort_values(ascending=False)

        print(sorted_corr)

        if plot:
            fig = go.Figure(data=go.Heatmap({'z': corr_all.values.tolist(),
                                             'x': corr_all.columns.tolist(),
                                             'y': corr_all.index.tolist()},
                                            hoverongaps=True))
            fig.show()

    def merger_weather_to_target(self, my_inp, areas=None):
        # - 4개 관측소의 시간별 기상 데이터를 입력받고, 4개 관측소의 기온 데이터의 중간값을 대표 기상 데이터로 활용
        # - 시간별 기온 데이터에 대한 결측치 대체 및 일별 데이터로 변환 (일별 최대값, 최소값, 중간값, 일교차 등)
        start = my_inp.data_start
        end = my_inp.data_end

        # 선택된 area의 날씨만 self.weather로 저장
        self._merge_weather(areas)

        # 결측치 처리
        self._fillna_weather(['temp'])

        weather = self.weather

        # 지역별 기상 데이터 처리
        weather_list = []
        for area in weather['area'].unique():
            weather_list.append(weather[weather['area'] == area].copy())

        for i, area in enumerate(weather['area'].unique()):
            weather_list[i].drop(['area'], axis=1, inplace=True)
            weather_list[i].columns = ['datetime'] + [str(area) + '_' + str(col) for col in weather.columns[2:]]

        # 시간별 기상 속성의 중간값 연산
        df_list = []

        i = 1
        for col in ['temp']:
            # print(col)
            hourly_df = pd.DataFrame(columns=['datetime'])
            date_range = pd.date_range(start, end, freq='H')
            hourly_df['datetime'] = date_range

            for d in weather_list:
                hourly_df = hourly_df.merge(d.iloc[:, [0, i]], how='outer')
            hourly_df['median'] = hourly_df.iloc[:, :len(weather_list) + 1].median(axis=1)
            hourly_df = hourly_df.loc[:, ['datetime', 'median']]
            hourly_df.columns = ['date', col]
            df_list.append(hourly_df)
            i = i + 1

        hourly_data = df_list[0].copy()
        for df in df_list[1:]:
            hourly_data = hourly_data.merge(df, how='left')

        # 결측치 처리
        hourly_data = hourly_data.fillna(method='bfill')

        # 일별 데이터로 변환(최대값, 최소값, 평균값, 일교차 등)
        daily_df = pd.DataFrame(columns=['date'])
        date_range = pd.date_range(start, end, freq='D')
        daily_df['date'] = date_range

        for col in ['temp']:
            for d in range(daily_df.shape[0]):
                for h in range(24):
                    daily_df.loc[d, str(col) + '_h' + str(h)] = hourly_data[col][d * 24:d * 24 + 24][d * 24 + h]

            daily_df[str(col) + '_max'] = daily_df.loc[:, str(col) + '_h0':].max(axis=1)
            daily_df[str(col) + '_min'] = daily_df.loc[:, str(col) + '_h0':].min(axis=1)
            daily_df[str(col) + '_mean'] = daily_df.loc[:, str(col) + '_h0':].mean(axis=1)
            daily_df[str(col) + '_range'] = daily_df[str(col) + '_max'] - daily_df[str(col) + '_min']
            daily_df[str(col) + '_ratio'] = abs(daily_df[str(col) + '_max'] - daily_df[str(col) + '_mean']) / daily_df[
                str(col) + '_range']

        target_data = self.target.copy()
        for col in ['temp']:
            target_data[str(col) + '_max'] = daily_df[str(col) + '_max']
            target_data[str(col) + '_min'] = daily_df[str(col) + '_min']
            target_data[str(col) + '_mean'] = daily_df[str(col) + '_mean']
            target_data[str(col) + '_range'] = daily_df[str(col) + '_range']
            target_data[str(col) + '_ratio'] = daily_df[str(col) + '_ratio']

        target_data['train_test'] = 'train'
        target_data = utils.downcast_dtypes(target_data)

        # TODO 임시
        date_range_cos = pd.to_numeric(date_range.strftime('%j'))
        a = np.array(date_range_cos)
        # tmp_cos = np.cos(4 * math.pi * (a-30) / 365)
        tmp_cos = np.cos(2 * math.pi * (a) / 365)
        target_data['date_cos'] = tmp_cos

        self.target = target_data

    def add_test_to_target(self, user_inp):
        start_date = datetime.strptime(user_inp.data_end, "%Y-%m-%d") + timedelta(days=1)
        end_date = datetime.strptime(user_inp.pred_end, "%Y-%m-%d")

        test_df = pd.DataFrame(columns=['date'])
        date_range = pd.date_range(start_date, end_date, freq='D')
        test_df['date'] = date_range

        test_df['smp_max'] = 0
        test_df['smp_min'] = 0
        test_df['smp_mean'] = 0
        test_df['supply'] = 0

        test_df['temp_max'] = 0
        test_df['temp_min'] = 0
        test_df['temp_mean'] = 0

        test_df['temp_range'] = 0
        test_df['temp_ratio'] = 0

        test_df['train_test'] = 'test'

        # TODO 임시
        date_range_cos = pd.to_numeric(date_range.strftime('%j'))
        a = np.array(date_range_cos)
        # tmp_cos = np.cos(4 * math.pi * (a - 30) / 365)
        tmp_cos = np.cos(2 * math.pi * (a) / 365)
        test_df['date_cos'] = tmp_cos

        self.target = pd.concat([self.target, test_df], axis=0)
        self.target.columns = test_df.columns
        self.target = self.target.reset_index().rename(columns={'index': 'num'})

    def add_time_features(self, is_crawling=True):
        tmp = self.target.copy()

        tmp['date'] = pd.to_datetime(tmp['date'])
        tmp['year'] = tmp['date'].dt.year
        tmp['month'] = tmp['date'].dt.month
        tmp['day'] = tmp['date'].dt.day
        tmp['dayofweek'] = tmp['date'].dt.dayofweek
        tmp['weekend'] = tmp['dayofweek'].map({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1})

        holiday = utils.hoilday_info(AUTH_KEY, is_crawling=is_crawling)

        tmp['holiday'] = tmp['date'].isin(pd.to_datetime(holiday['date'])).astype(int)
        tmp['is_holiday'] = (tmp['weekend'] + tmp['holiday']).map({0: 0, 1: 1, 2: 1})
        tmp = tmp.drop(['weekend', 'holiday'], axis=1)

        self.target = tmp

    def add_tourist_info(self, is_crawling=True):
        tourist = utils.tourist_info(is_crawling=is_crawling)

        self.target = self.target.merge(tourist, on=['year', 'month'], how='left')
        self.target = self.target.replace(np.inf, np.nan).fillna(0)

    def add_energy_cost(self):
        energy_cost = self.energy_cost
        energy_cost = energy_cost.sort_values('기간').reset_index(drop=True)
        energy_cost.loc[28] = ['2020/06'] + energy_cost.iloc[27, 1:].to_list()
        energy_cost.iloc[:, 1:] = energy_cost.iloc[:, 1:].astype(float)
        energy_cost.columns = ['time', 'nuclear', 'coal1', 'coal2', 'oil', 'gas']
        energy_cost[['year', 'month']] = energy_cost['time'].str.split('/', expand=True)
        energy_cost[['year', 'month']] = energy_cost[['year', 'month']].astype(int)
        energy_cost = energy_cost.drop(['time'], axis=1)

        self.target = self.target.merge(energy_cost, on=['year', 'month'], how='left')
        self.target = self.target.replace(np.inf, np.nan).fillna(0)

    def one_hot_encoding(self, cols):
        dataset = self.target.copy()

        for col in cols:
            onehot = pd.get_dummies(self.target[col], prefix=col)
            dataset = dataset.join(onehot)

        self.target = dataset.drop(cols, axis=1)

    def plot_all_data(self):
        data = self.target.loc[:, 'date':'temp_ratio']
        cols = len(data.columns) - 1
        fig = make_subplots(rows=cols, cols=1)

        for i in range(cols):
            fig.append_trace(go.Scatter(
                x=data['date'],
                y=data.iloc[:, i+1],
                name=data.columns.values[i+1]
            ), row=i+1, col=1)

        fig.show()

    def add_pred_result(self, pred_feature, pred_data):
        test_idx = self.target.loc[(self.target.num == 0) & (self.target.train_test == 'test'), :].index[0]
        target_cols = ['temp_max', 'temp_min', 'temp_mean', 'temp_range', 'temp_ratio']

        if pred_feature == 'temperature':
            target_cols = ['temp_max', 'temp_min', 'temp_mean', 'temp_range', 'temp_ratio']
        elif pred_feature == 'supply':
            target_cols = 'supply'
        elif pred_feature == 'smp':
            target_cols = ['smp_max', 'smp_min', 'smp_mean']

        self.target.loc[test_idx:, target_cols] = pred_data.loc[test_idx:, target_cols]

    def export_submission_data(self):
        pred_start, pred_end = self.user_inp.pred_start, self.user_inp.pred_end
        hist_start, hist_end = self.user_inp.data_start, self.user_inp.data_end

        pred_cols = ['smp_max', 'smp_min', 'smp_mean', 'supply']

        history_mask = (self.target['date'] >= hist_start) & (self.target['date'] <= hist_end)
        history_df = self.target.loc[history_mask, pred_cols]

        pred_mask = (self.target['date'] >= pred_start) & (self.target['date'] <= pred_end)
        pred_df = self.target.loc[pred_mask, pred_cols]

        real_df_tmp = pd.read_csv(os.path.join(self.data_dir, 'real_data.csv'))
        real_mask = (real_df_tmp['date'] >= pred_start) & (real_df_tmp['date'] <= pred_end)
        real_df = real_df_tmp.loc[real_mask, pred_cols]

        pred_df.to_csv(os.path.join(self.base_dir, 'output\\pred_df.csv'))
        real_df.to_csv(os.path.join(self.base_dir, 'output\\real_df.csv'))

        score = utils.calc_rmsse(real_df, pred_df, history_df, axis=0, weight=[0.1, 0.1, 0.2, 0.6])
        print(score)

        pred_df.columns = ['smp_max_pred', 'smp_min_pred', 'smp_mean_pred', 'supply_pred']
        real_df.columns = ['smp_max_real', 'smp_min_real', 'smp_mean_real', 'supply_real']

        plot_df1 = pred_df.join(real_df)
        plot_df1.index = pd.date_range(pred_start, pred_end)

        fig1 = px.line(plot_df1.astype(float), x=plot_df1.index, y=plot_df1.columns[:], line_shape='linear')

        fig1.update_xaxes(title_text='Time')
        fig1.update_traces(hovertemplate=None)
        fig1.update_layout({'legend_title_text': ''}, hovermode="x unified")

        fig1.write_html(os.path.join(self.base_dir, 'output\\Result_plot.html'))
        fig1.show()

    def _merge_weather(self, area_nums):
        # 4개 관측소 데이터만 선택하여 정리하는 함수 (제주, 고산, 성산, 서귀포)

        weather = self.weather
        areas = area_nums

        weather_list = []
        for area in areas:
            weather_list.append(weather.loc[weather['area'] == area, :])

        result = pd.concat(weather_list, axis=0)

        self.weather = result

    def _fillna_weather(self, cols):
        # 기상 데이터에 대한 결측값을 보간법 및 1시간 이후 데이터로 대체하는 함수
        #  - QCFlag 종류: 0(정상), 1(오류), 9(결측)

        weather = self.weather

        weather_features = ['area', 'datetime'] + cols + [col + '_QCFlag' for col in cols]
        weather_data = weather.loc[:, weather_features]
        weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
        weather_data = weather_data.set_index('datetime')

        for col in cols:
            weather_data.loc[weather_data[col + '_QCFlag'] == 9.0, col] = np.nan
            weather_data.loc[weather_data[col + '_QCFlag'] == 1.0, col] = np.nan

            weather_data[col] = weather_data[col].interpolate(method='time')  # 보간법 활용 결측치 처리
            weather_data[col] = weather_data[col].fillna(method='bfill')  # 1시간 이후 데이터로 결측치 처리
            weather_data = weather_data.drop([col + '_QCFlag'], axis=1)

        weather_data = weather_data.reset_index(drop=False)
        weather_data = utils.downcast_dtypes(weather_data)

        self.weather = weather_data
