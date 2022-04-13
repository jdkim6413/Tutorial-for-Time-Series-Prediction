import numpy as np
import random
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow import keras
import tensorflow as tf

import os
import json

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import keras_tuner as kt


# 시드 고정
SEED = 25
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class PredModel:
    def __init__(self, pred_feature, output_dir):
        self.pred_feature = pred_feature

        self.train_features = None  # 학습에 사용할 Featrues
        self.target_features = None   # 예측할 Featrues

        self.history_size = 7  # 몇일동안 데이터를 가지고 다음을 예측할 지
        self.target_size = 1  # 한번에 몇일을 예측할 지
        self.val_days = 30  # 검증 데이터 수
        self.step = 1

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

        self.data_for_pred = None
        self.scaler = None

        self.model = None
        self.history = None

        self.lags = None
        self.windows = None

        if pred_feature == 'temperature':
            # self.train_features = ['temp_max', 'temp_min', 'temp_mean', 'temp_range', 'temp_ratio',
            #                        'month', 'day', 'year_2018', 'year_2019', 'year_2020']
            self.train_features = ['temp_max', 'temp_min', 'temp_mean', 'temp_range', 'temp_ratio',
                                   'date_cos']
            self.target_features = ['temp_max', 'temp_min', 'temp_mean']
            self.lags = [1, 2]  # [1, 2]
            self.windows = [3, 5]  # [3, 5]
        elif pred_feature == 'supply':
            # self.train_features = ['supply', 'temp_max', 'temp_min', 'temp_mean', 'temp_range', 'temp_ratio',
            #                        'month', 'day', 'dayofweek', 'tourist', 'year_2018', 'year_2019', 'year_2020',
            #                        'is_holiday_0', 'is_holiday_1']
            self.train_features = ['supply', 'temp_max', 'temp_min', 'temp_mean', 'temp_range', 'temp_ratio',
                                   'dayofweek', 'tourist',
                                   'is_holiday_0', 'is_holiday_1', 'date_cos']
            self.target_features = ['supply']
            self.lags = [1, 2]  # [1, 2, 3, 7]
            self.windows = [3, 5]  # [5, 10]
        elif pred_feature == 'smp':
            # self.train_features = ['smp_max', 'smp_min', 'smp_mean', 'supply', 'temp_max', 'temp_min', 'temp_mean',
            #                        'month', 'day', 'dayofweek', 'temp_range', 'temp_ratio', 'tourist', 'coal1',
            #                        'oil', 'gas', 'year_2018', 'year_2019', 'year_2020', 'is_holiday_0', 'is_holiday_1']
            self.train_features = ['smp_max', 'smp_min', 'smp_mean', 'supply', 'temp_max', 'temp_min', 'temp_mean',
                                   'dayofweek', 'temp_range', 'temp_ratio', 'tourist', 'coal1',
                                   'oil', 'gas', 'is_holiday_0', 'is_holiday_1', 'date_cos']
            self.target_features = ['smp_max', 'smp_min', 'smp_mean']
            self.lags = [1, 2]  # [1, 2, 3]
            self.windows = [3, 5]  # [3, 5, 7]
        else:
            print('error')

        self.model_dir = os.path.join(output_dir, pred_feature)

    def transform_dataset(self, target_data, start_idx=0):
        temp_train, temp_valid = self._process_data_for_train(target_data)

        history_size = self.history_size
        target_size = self.target_size
        step = self.step

        train_cols = self.train_features
        target_cols = []
        for col_names in self.target_features:
            target_cols.append(col_names + '_target')

        for cur_data in ['train', 'valid']:
            if cur_data == 'train':
                dataset = temp_train
            else:
                dataset = temp_valid

            dataset.index = range(dataset.shape[0])

            data = []
            labels = []

            end_index = len(dataset) - target_size + 1

            for i in range(start_idx + history_size, end_index):
                indices = range(i - history_size, i, step)
                indices = np.array(dataset.loc[indices, :][train_cols])
                data.append(indices)

                if target_size == 1:
                    labels.append(np.array(dataset.loc[i + target_size - 1, target_cols]))
                elif target_size >= 2:
                    labels.append(np.array(dataset.loc[i:i + target_size - 1, target_cols]))
                else:
                    print("Error")

            data = np.array(data).astype(np.float32)

            try:
                labels = np.array(labels).astype(np.float32)
            except:
                labels = np.concatenate(labels, axis=1).astype(np.float32)

            if cur_data == 'train':
                self.x_train = data
                self.y_train = labels
            else:
                self.x_val = data
                self.y_val = labels

    def create_model(self, epochs=500, batch_size=16, buffer_size=1000, is_fromfile=False, is_tune_para=False):
        if not is_fromfile:
            x_train = self.x_train
            y_train = self.y_train
            x_val = self.x_val
            y_val = self.y_val

            tf.random.set_seed(SEED)

            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).prefetch(buffer_size)

            val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_data = val_data.batch(batch_size).prefetch(buffer_size)

            if is_tune_para:
                tuner = kt.RandomSearch(self._conv_lstm_hyper_model,
                                        objective='val_loss',
                                        max_trials=32,
                                        directory=os.path.join(self.model_dir, 'kerastuner'),
                                        project_name='optimized')

                tuner.search(train_data, epochs=10, validation_data=val_data)
                tuner.reload()

                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                model = tuner.hypermodel.build(best_hps)
            else:
                n_filter, dense1, dense2, dense3, lr = (64, 256, 64, 20, 1.2e-5)
                if self.pred_feature == 'supply':
                    n_filter, dense1, dense2, dense3, lr = (30, 256, 64, 20, 6.3e-6)
                elif self.pred_feature == 'smp':
                    n_filter, dense1, dense2, dense3, lr = (60, 128, 32, 10, 2.31e-5)

                model = self._conv_lstm_model(n_filter, dense1, dense2, dense3, lr=lr)

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=20,
                                                              restore_best_weights=True)

            history = model.fit(train_data,
                                epochs=epochs,
                                validation_data=val_data,
                                callbacks=[early_stopping])

            # result = model.evaluate(val_data)
            # print(dict(zip(model.metrics_names, result)))

            model.fit(val_data, epochs=30, verbose=0)
            print(model.evaluate(val_data))

            self.history = history.history
            self.model = model
            self._save_model()
        else:
            self._load_model()

    def predict_feature(self, target_data, n_days=50):
        history_size = self.history_size
        target_size = self.target_size
        test_split = target_data['train_test'].value_counts()[0]  # num of train data
        n_features = len(self.train_features)

        target_index = target_data.loc[(target_data.num == 0) & (target_data.train_test == 'test'), :].index[0]
        pred_data = self.data_for_pred.loc[test_split - history_size - target_size:, self.train_features]

        for day_num in range(0, n_days):
            test_data = pred_data.loc[target_index + day_num - history_size:target_index + day_num - 1, :]
            test_data_scaled = self.scaler.transform(test_data)
            scaled_inp = test_data_scaled[:, :].reshape(1, history_size, n_features)
            pred = self.model.predict(scaled_inp)[0]
            pred_data.loc[target_index + day_num, self.target_features] = pred

            if self.pred_feature == "temperature":
                temp_max = pred_data.loc[target_index + day_num, 'temp_max']
                temp_min = pred_data.loc[target_index + day_num, 'temp_min']
                temp_mean = pred_data.loc[target_index + day_num, 'temp_mean']
                temp_range = temp_max - temp_min

                pred_data.loc[target_index + day_num, 'temp_range'] = temp_range
                pred_data.loc[target_index + day_num, 'temp_ratio'] = abs(temp_max - temp_mean) / temp_range

            target_feat = self.target_features

            if self.pred_feature == "supply":
                target_feat = ['supply', 'temp_mean']

            lag_idx = history_size + pred_data.index[0]

            for col in target_feat:
                # lag features
                for lag in self.lags:
                    tmp_lag = pred_data[col].shift(lag)
                    pred_data.loc[lag_idx:, col + '_lag_%s' % lag] = tmp_lag.loc[lag_idx:]
                    # pred_data[col + '_lag_%s' % lag] = pred_data[col].shift(lag)
                # trend features
                for window in self.windows:
                    tmp_mean = pred_data[col].shift(1).rolling(window, min_periods=1).mean()
                    pred_data.loc[lag_idx:, col + '_ma_mean_%s' % window] = tmp_mean.loc[lag_idx:]

                    tmp_std = pred_data[col].shift(1).rolling(window, min_periods=1).std()
                    pred_data.loc[lag_idx:, col + '_ma_std_%s' % window] = tmp_std.loc[lag_idx:]

            pred_data = pred_data.replace([np.inf, -np.inf], np.nan)
            pred_data = pred_data.fillna(pred_data.mean())

        pred_data.to_csv("pred_data.csv")
        return pred_data

    def _process_data_for_train(self, target_data, start=0):
        """
        - 데이터셋에 대한 Feature Enineering 함수
        - lag features : lag 속성
        - trend features : 이동평균의 평균 및 표준편차
        """

        history_size = self.history_size
        val_days = self.val_days

        tmp = target_data.copy()

        cols = self.train_features
        test_split = tmp['train_test'].value_counts()[0]  # num of train data
        train_split = test_split - val_days - history_size

        target_feat = self.target_features

        if self.pred_feature == "supply":
            target_feat = ['supply', 'temp_mean']

        for col in target_feat:
            # lag features
            for lag in self.lags:
                tmp[col + '_lag_%s' % lag] = tmp[col].shift(lag)
                cols.append(col + '_lag_%s' % lag)
            # trend features
            for window in self.windows:
                tmp[col + '_ma_mean_%s' % window] = tmp[col].shift(1).rolling(window, min_periods=1).mean()
                tmp[col + '_ma_std_%s' % window] = tmp[col].shift(1).rolling(window, min_periods=1).std()
                cols.append(col + '_ma_mean_%s' % window)
                cols.append(col + '_ma_std_%s' % window)

        tmp = tmp.replace([np.inf, -np.inf], np.nan)
        tmp = tmp.fillna(tmp.mean())

        train = tmp.loc[start:train_split - 1, :]
        valid = tmp.loc[train_split:test_split - 1, :]

        # scaler = MinMaxScaler()
        # scaler = RobustScaler()
        scaler = StandardScaler()

        train_scaled = scaler.fit_transform(train[cols])
        valid_scaled = scaler.transform(valid[cols])

        for col in target_feat:
            train[col + '_target'] = train[col]
            valid[col + '_target'] = valid[col]

        train[cols] = train_scaled
        valid[cols] = valid_scaled

        self.train_features = cols
        self.scaler = scaler
        self.data_for_pred = tmp

        return train, valid

    def _save_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.chdir(self.model_dir)
        model = self.model

        # save model
        model.save(self.pred_feature + "_model")

        # save history
        json.dump(self.history, open(self.pred_feature + '_history.json', 'w'))
        self._plot_history()
        print('\n Model Saved')

    def _load_model(self):
        os.chdir(self.model_dir)

        # load model
        loaded_model = models.load_model(self.pred_feature + "_model")
        self.model = loaded_model

        # load history
        self.history = json.load(open(self.pred_feature + '_history.json', 'r'))
        print('\n Model Loaded')

    def _plot_history(self, is_plot=False):
        history = self.history

        print('\n-------- Plotting Process (Accuracy & Loss) --------')
        plt.rcParams["figure.figsize"] = (12, 4)
        plt.rcParams.update({'font.size': 12})
        plt.figure()
        plt.style.use('default')
        plt.subplot(121)
        plt.title('Loss Plot')
        plt.plot(history['loss'], label='train loss')
        plt.plot(history['val_loss'], label='val loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.subplot(122)
        plt.title('MAE Plot')
        plt.plot(history['mae'], label='train MAE')
        plt.plot(history['val_mae'], label='val MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig('Training_history.png', dpi=360)
        if is_plot:
            plt.show()
        plt.clf()
        plt.close('all')

    def _conv_lstm_hyper_model(self, hp):
        n_filter = hp.Choice('n_filter', [32, 64])
        dense1 = hp.Choice('dense1', [128, 256])
        dense2 = hp.Choice('dense2', [32, 64])
        dense3 = hp.Choice('dense3', [10, 20])

        base_model = self._conv_lstm_model(n_filter, dense1, dense2, dense3)

        return base_model

    def _conv_lstm_model(self, n_filter, dense1, dense2, dense3, lr=1.2e-5):
        n_filter = n_filter
        dense1 = dense1
        dense2 = dense2
        dense3 = dense3

        n_features = len(self.train_features)
        n_target = len(self.target_features)

        keras.backend.clear_session()
        tf.random.set_seed(SEED)

        # Conv1D + 양방향 LSTM + DNN 결합 모델로 훈련
        ts_input = keras.Input(shape=(None, n_features), name='ts_input')
        x = Conv1D(filters=n_filter, kernel_size=5, strides=1, padding="causal", activation="relu")(ts_input)

        # x = Bidirectional(LSTM(128, recurrent_dropout=0.2, dropout=0.2, return_sequences=True))(x)
        # x = Bidirectional(LSTM(128, recurrent_dropout=0.5, dropout=0.2, return_sequences=False))(x)
        # x = Dense(dense1, activation="relu")(x)
        # x = Dropout(0.2)(x)
        # x = Dense(dense2, activation="relu")(x)
        # x = Dropout(0.2)(x)
        # x = Dense(dense3, activation="relu")(x)
        # x = Dense(n_target)(x)

        if self.pred_feature == 'temperature':
            x = Bidirectional(LSTM(128, recurrent_dropout=0.2, dropout=0.2, return_sequences=True))(x)
            x = Bidirectional(LSTM(128, recurrent_dropout=0.5, dropout=0.2, return_sequences=False))(x)
            x = Dense(dense1, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(dense2, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(dense3, activation="relu")(x)
            x = Dense(n_target)(x)
        elif self.pred_feature == 'supply':
            x = Bidirectional(LSTM(128, recurrent_dropout=0.2, dropout=0.2, return_sequences=True))(x)
            x = Bidirectional(LSTM(128, recurrent_dropout=0.5, dropout=0.2, return_sequences=False))(x)
            x = Dense(dense1, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(dense2, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(128, activation="relu")(x)
            x = Dense(dense3, activation="relu")(x)
            x = Dense(n_target)(x)
        elif self.pred_feature == 'smp':
            x = Bidirectional(LSTM(128, dropout=0.2, return_sequences=True))(x)
            x = Bidirectional(LSTM(128, recurrent_dropout=0.5, dropout=0.2, return_sequences=False))(x)
            x = Dense(dense1, activation="relu")(x)
            x = Dense(64, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(dense2, activation="relu")(x)
            x = Dense(dense3, activation="relu")(x)
            x = Dense(n_target)(x)

        output = Lambda(lambda x: x * 200, name='output')(x)

        model = keras.Model(inputs=[ts_input], outputs=[output])

        optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])

        return model
