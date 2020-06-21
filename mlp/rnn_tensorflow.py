import tensorflow as tf
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
import params as params
import time

df = pd.read_csv('preproc.csv')
TRAIN_SPLIT = round(df.shape[0]*0.8)
tf.random.set_seed(13)

# Start time
def get_datetime():
    print('')
    print('---------- ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ----------')
    print('')

def scale_attribute(data, attribute=None):
    ss_data = data

    if attribute != None:
        ss_data = ss_data[attribute]
    else:
        ss_data = ss_data.drop(['date_hr'], axis=1, errors='ignore')
    ss_data.index = data['date_hr']
    
    # Scaling features before training a neural network
    if attribute != None:
        ss_train_mean = ss_data[:TRAIN_SPLIT].mean()
        ss_train_std = ss_data[:TRAIN_SPLIT].std()
    else:
        ss_train_mean = ss_data[:TRAIN_SPLIT].mean(axis=0)
        ss_train_std = ss_data[:TRAIN_SPLIT].std(axis=0)

    # Standardize Data
    ss_data = (ss_data-ss_train_mean)/ss_train_std
    return ss_data.values

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

def multi_step_plot(history, true_future, prediction, title):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.title(title)
    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/params.STEP, np.array(true_future), 'bo',
            label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/params.STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    
    plt.legend(loc='upper left')
    return plt

def create_time_steps(length):
    return list(range(-length, 0))

def baseline(history):
    return np.mean(history)

def main():
    get_datetime()

    if params.command_2 == 'univariate':
        univariate_past_history = params.uni_past_history
        univariate_future_target = params.uni_future_target
        ss_preproc = scale_attribute(df, params.attribute)

        x_train_uni, y_train_uni = univariate_data(ss_preproc, 0, TRAIN_SPLIT,
                                                   univariate_past_history,
                                                   univariate_future_target)
        x_val_uni, y_val_uni = univariate_data(ss_preproc, TRAIN_SPLIT, None,
                                               univariate_past_history,
                                               univariate_future_target)

        # Generate Baseline Plot
        base_plot = show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
                             'Baseline Prediction for '+ params.attribute)

        base_plot.savefig('mlp/img/univariate_baseline_predict_' + params.attribute + '_'
                           + datetime.now().strftime("%Y-%m-%d_%Hhr_%Mm_%Ss") + '.png')
        base_plot.clf()
        print('Baseline prediction lineplot generated for '+ params.attribute)

        # Training Simple LSTM Model
        train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
        train_univariate = train_univariate.cache().shuffle(params.BUFFER_SIZE).batch(params.BATCH_SIZE).repeat()
        val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
        val_univariate = val_univariate.batch(params.BATCH_SIZE).repeat()

        simple_lstm_model = tf.keras.models.Sequential([
                                    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
                                    tf.keras.layers.Dense(1)
                                    ])
        simple_lstm_model.compile(optimizer='adam', loss='mae')

        simple_lstm_model.fit(train_univariate, epochs=params.EPOCHS,
                              steps_per_epoch=params.EVALUATION_INTERVAL,
                              validation_data=val_univariate, validation_steps=50)
        
        for x, y in val_univariate.take(3):
            
            plot = show_plot([x[0].numpy(), y[0].numpy(),
                            simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM prediction model for ' + params.attribute) 
            plot.savefig('mlp/img/univariate_simple_lstm_predict_' + params.attribute + '_' +
                        datetime.now().strftime("%Y-%m-%d_%Hhr_%Mm_%Ss") + '.png')
            plot.clf()
            time.sleep(2)
    
    if params.command_2 == 'multivariate':
        multi_past_history, multi_future_target = params.multi_past_history, params.multi_future_target
        STEP = params.STEP 
        column_ls = [x for x in df.columns]
        
        # Attribute to Predict
        target = column_ls.index(params.attribute) 
        
        ss_preproc = scale_attribute(df)

        x_train_multi, y_train_multi = multivariate_data(ss_preproc, ss_preproc[:, target], 0,
                                                 TRAIN_SPLIT, multi_past_history,
                                                 multi_future_target, STEP)
        x_val_multi, y_val_multi = multivariate_data(ss_preproc, ss_preproc[:, target],
                                                    TRAIN_SPLIT, None, multi_past_history,
                                                    multi_future_target, STEP)
        # Training Multi Model LSTM Model
        train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
        train_data_multi = train_data_multi.cache().shuffle(params.BUFFER_SIZE).batch(params.BATCH_SIZE).repeat()

        val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
        val_data_multi = val_data_multi.batch(params.BATCH_SIZE).repeat()

        multi_step_model = tf.keras.models.Sequential()
        multi_step_model.add(tf.keras.layers.LSTM(32,
                                                return_sequences=True,
                                                input_shape=x_train_multi.shape[-2:]))
        multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
        multi_step_model.add(tf.keras.layers.Dense(72))

        multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

        multi_step_history = multi_step_model.fit(train_data_multi, epochs= params.EPOCHS,
                                                steps_per_epoch= params.EVALUATION_INTERVAL,
                                                validation_data= val_data_multi,
                                                validation_steps=50)
        
        for x, y in val_data_multi.take(3):
            plot = multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0], 'Muti-Variate LSTM prediction model for ' + params.attribute)
            plot.savefig('mlp/img/multivariate_lstm_predict_' + params.attribute + '_' +
                        datetime.now().strftime("%Y-%m-%d_%Hhr_%Mm_%Ss") + '.png')
            plot.clf()
            time.sleep(2)

    print('Job Completed!')
    get_datetime()
    
if __name__ == '__main__':
    main()

