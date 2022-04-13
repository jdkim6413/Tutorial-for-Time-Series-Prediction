#!/usr/bin/env python
# coding: utf-8

import userinput as inp
import dataset
import predmodel

import warnings
warnings.filterwarnings(action='ignore')


# User Input
user_inp = inp.UserInput(data_start='2018-02-01',
                         data_end='2020-05-18',
                         pred_start='2020-05-19',
                         pred_end='2020-06-08',
                         drive_path='D:\\',  ##  input drive path
                         area=[184, 185, 188, 189],
                         is_crawling=True,
                         is_modelfromfile=True,
                         is_tune_para=False)
# drive_path='./')

my_data = dataset.DataSet(user_inp)
my_data.read_data()

# ## Data Pre-Processing  & Feature Engineering
# 관측소 선정 (제주, 고산, 성산, 서귀포 4개관측소 선정)
#   - 선정이유: 기상 속성이 다양한 ASOS / 결측값이 적은곳 / 동서남북 대표 1개소
# 기상속성 선정 (기온 하나만 선정)
#   - 선정 이유: 결측값이 적은 속성, SMP 및 supply와의 상관관계 높음
my_data.show_asos_area()
my_data.show_missing_value(plot=False)
my_data.show_avg_correlation(plot=False)
selected_areas = user_inp.area  # 제주, 고산, 성산, 서귀포

# 기온 데이터 전처리
# - 4개 관측소 기온 데이터의 중간값(median)을 활용
# - 시간별 기온 데이터 중에서 결측치를 보간법 및 1시간 뒤의 기온 데이터로 대체
# - 시간별 기온 데이터를 일별 데이터로 변환하고 속성을 추가
# - 타겟 데이터와 결합하여 훈련 데이터셋 준비
my_data.merger_weather_to_target(user_inp, areas=selected_areas)  # 날씨속성을 target df에 추가

my_data.add_test_to_target(user_inp)  # target data에 test data(예측할 데이터) dummy 행 추가
my_data.add_time_features(is_crawling=user_inp.is_crawling)  # Time column에 날짜정보 및 공휴일 정보 추가
my_data.add_tourist_info(is_crawling=user_inp.is_crawling)  # 제주특별자치도 정보공개게시판의 관광객 입도 현황(90) 자료를 크롤링
my_data.add_energy_cost()  # 연료비 단가 추가
my_data.one_hot_encoding(['year', 'is_holiday'])  # One Hot Encoding (year, is_holiday)
# my_data.plot_all_data()  # Data plot

pred_feature = ['temperature', 'supply', 'smp']
for feature in pred_feature:
    cur_model = predmodel.PredModel(feature, user_inp.output_path)
    cur_model.transform_dataset(my_data.target)

    cur_model.create_model(is_fromfile=user_inp.is_modelfromfile,
                           is_tune_para=user_inp.is_tune_para)

    pred_data = cur_model.predict_feature(my_data.target)

    my_data.add_pred_result(feature, pred_data)

my_data.target.to_csv(user_inp.output_path + 'self.targetdata.csv')
my_data.export_submission_data()
