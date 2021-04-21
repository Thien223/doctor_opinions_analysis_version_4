
import tensorflow as tf
from sklearn.externals import joblib
import copy
from keras.utils.training_utils import multi_gpu_model
import keras.backend as K
import keras.optimizers as optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
from config import hparams as config
from utils import *
import datetime


### because the data has missing data from 2018-10-31 to 2018-12, then we have to cut out missing part.
def update_data():
    building_name_to_building_id = pd.read_csv('building_name_to_building_id.csv', header=None)
    for row in building_name_to_building_id.itertuples():
        try:
            new_data, building = loadData('data/{}.csv'.format(str(row[1])), freq='15T')
            old_data, _ = loadData('data/old/{}.csv'.format(str(row[2])), freq='15T')
            old_data.index=[idx.replace(year=idx.year+1) for idx in old_data.index]
            if building == 'B0014':
                old_data = old_data.loc['2018-08-21':]
            else:
                old_data = old_data.loc['2018-10-31':]
            data = old_data.combine_first(new_data)
            data.to_csv('new_data/{}.csv'.format(str(row[1])))
        except Exception as e:
            print(e)
            pass

def train(_link):
    raw_data, model_name = loadData(_link, freq='15T')
    _data = raw_data.resample('15T').interpolate()
    _data = _data.resample('15T').sum().reindex(pd.date_range(_data.index[0], _data.index[-1],freq='15T'))  ### convert data to 15 minutes frequency, using sum() function
    temp_df = copy.deepcopy(_data)
    # plt.plot(temp_df)
    # temp_df = temp_df.loc[:'2018-08-20',:]
    ### shift index to next 2 days
    old_index = temp_df.index
    new_index = old_index + datetime.timedelta(minutes=15*config.time_steps)
    temp_df = temp_df.set_index(new_index)
    ## add temperature and weekday_index (of next 2 days)
    new_data = add_hourly_temp_column_from_csv(temp_df)
    data_ = add_weekday_column(new_data)

    ##set index back to original one
    data_ = data_.set_index(old_index)



    # train_data, test_data = splitData(data_, trainPercent=0.92)
    train_data = copy.deepcopy(data_) ### use all data as training set


    train_data_scaled, scaler = scale_to_0and1(train_data)
    joblib.dump(scaler, 'pretrained_models/linear_activation/{}.scaler'.format(model_name))

    ### preparing lstm input and output sequences
    train_input, train_output = sequentialize_multiple_features_data(scaled_inputData=train_data_scaled, inputData_index= train_data.index, output_indexes=[0], config=config)

    #construct model
    K.clear_session()
    model = Sequential()
    model.add((LSTM(units=config.num_units,input_shape=(config.time_steps, config.features))))  # (timestep, feature)
    model.add(Activation('linear'))
    model.add(Dense(units=1))
    # model = multi_gpu_model(model, gpus=2)
    optimizer = optimizers.Adam()
    model.compile(loss="mse", optimizer=optimizer)
    ### print model summary, input output shape to debug
    model.summary()
    print("Inputs: {}".format(model.input_shape))
    print("Outputs: {}".format(model.output_shape))
    print("Actual input: {}".format(train_input.shape))
    print("Actual output: {}".format(train_output.shape))
    ### define early stopping object
    early_stop = EarlyStopping(monitor="loss", patience=5, verbose=1)

    #### training with early stopping
    model.fit(train_input, train_output, epochs=config.epochs, batch_size=config.batch_size, verbose=2,callbacks=[early_stop])
    model.save('pretrained_models/linear_activation/{}.h5'.format(model_name))  ## save checkpoint




    import time

    #INFERENCE
    start = time.time()

    raw_data, model_name = loadData('data\\all.csv', freq='15T')


    _data = raw_data.resample('15T').interpolate()
    _data = _data.resample('15T').sum().reindex(pd.date_range(_data.index[0], _data.index[-1],
                                                              freq='15T'))  ### convert data to 15 minutes frequency, using sum() function
    temp_df = copy.deepcopy(_data)
    # plt.plot(temp_df)
    # temp_df = temp_df.loc[:'2018-08-20',:]
    ### shift index to next 2 days
    old_index = temp_df.index
    new_index = old_index + datetime.timedelta(minutes=15 * config.time_steps)
    temp_df = temp_df.set_index(new_index)
    ## add temperature and weekday_index (of next 2 days)
    new_data = add_hourly_temp_column_from_csv(temp_df)
    data_ = add_weekday_column(new_data)

    ##set index back to original one
    data_ = data_.set_index(old_index)

    # data__ = data_[:'2018-10-31']  ### cut out mising data part
    #
    # # ## split data
    train_data, test_data = splitData(data_, trainPercent=0.92)

    ### inference
    from keras.models import load_model


    predicted_building_nergy = pd.read_csv('jaja_2.csv')
    predicted_building_nergy['Unnamed: 0'] = pd.to_datetime(predicted_building_nergy['Unnamed: 0'])
    predicted_building_nergy.set_index('Unnamed: 0', inplace=True)
    predicted_building_nergy = predicted_building_nergy.resample('15T').interpolate()
    predicted_building_nergy = add_hourly_temp_column_from_csv(predicted_building_nergy)
    predicted_building_nergy = add_weekday_column(predicted_building_nergy)

    ### past temperature
    temp = pd.read_csv('temperature.csv')
    temp[temp.columns[0]] = pd.to_datetime(temp[temp.columns[0]])
    temp.set_index(temp.columns[0], inplace=True)
    # ## transform to 15 minutes frequency
    temp = temp.resample('15T').interpolate()


    ### define forecast length
    forecastLength = 96*1  ## 1 days
    forecasted = []
    ### split data into small piece (2304 points) and pass to API
    window_size = 673  ### 24 days
    start_index = 0
    with tf.Session(graph=K.get_session().graph) as session:
        session.run(tf.global_variables_initializer())
        ### inference
        model = load_model('pretrained_models/linear_activation/{}.h5'.format(model_name))
        scaler = joblib.load('pretrained_models/linear_activation/{}.scaler'.format(model_name))

        for i in range(int(len(predicted_building_nergy) / forecastLength)):
            #### initialize input window
            window = predicted_building_nergy[start_index:start_index + window_size]  ### get data from start index to 24days after
            # window = train_data
            next_index = list(window.index)
            window = scaler.transform(window)
            #### sequentializing input data
            input_, output_ = sequentialize_multiple_features_data(scaled_inputData=window, inputData_index=next_index, output_indexes=[0], config=config)
            start_index = start_index + forecastLength  ## move index to next part

            # ### check model performance
            # test_data_scaled = scaler.transform(test_data)
            # test_input, test_output = sequentialize_multiple_features_data(test_data_scaled, test_data.index,[0], config)
            # pred = model.predict(test_input)
            # plt.plot(test_output)
            # plt.plot(pred)
            for i in range(forecastLength):
                ## add next time index to old index list
                next_index.append(next_index[-1] + datetime.timedelta(minutes=15))
                ### define index of temperature and weekday (we take temperature and weekday of next 7days) 15*config.time_steps = 7days
                next_temp_and_weekday_index = next_index[-1] + datetime.timedelta(minutes=15*config.time_steps)
                ## predict
                predictedSequence = model.predict(input_, batch_size=int(len(window)-1))
                ## add result to result list
                forecasted.append(float(predictedSequence[-1]))
                #### preparing next prediction input####
                ########################################
                ## get temperature
                temperature_ = temp['Temperature'][str(next_temp_and_weekday_index)]
                ## get weekday index
                weekday_index_ = get_weekday_index(next_temp_and_weekday_index)
                ### add them to new row (we add 0 to array to make sure array has 3 columns)
                next_row_ = np.array([[0,temperature_, weekday_index_]])
                ## transform temperature and weekday index scale (the scaler require input has 3 columns)
                next_row_ = scaler.transform(next_row_)
                ## replace predicted value by original one
                next_row_[0][0] = predictedSequence[-1]
                ## expand dimension of row to 3D array [batch_size, time_steps, features] (batch_size=1)
                next_row = np.expand_dims(next_row_, axis=1)
                ## remove the last element and add predicted element into first possition
                input_ = np.concatenate((next_row,input_[:,:-1,:]), axis=1)
                print('step: {}'.format(i))



    ### transform predicted data to normal scale
    forecasted_np = np.array(forecasted).reshape(-1,1)
    ### add 2 columns (since scaler require 3 columns shape)
    forecasted_np_ = np.concatenate((forecasted_np, np.zeros(shape=(len(forecasted), 2))), axis=1)
    ## rescale to normal energy scale
    forecasted_np_rescaled = scaler.inverse_transform(forecasted_np_)
    ## add time index (since result has 3 columns, we just take the output column at the first possition)
    forecasted_df = pd.DataFrame(forecasted_np_rescaled[:,0], index=next_index[-forecastLength:])
    forecasted_df.to_csv('predicted_energy_2.csv')
    ### plot
    plt.title('Forecasted Energy Consumption - Building {}'.format(model_name))
    plt.plot(forecasted_df, label='Predicted')
    plt.plot(test_data['Power'][:forecastLength], label='Real unseen')
    plt.plot(train_data['Power'], label='Real seen')

    plt.legend()
    plt.grid()
    plt.show()
    end = time.time()
    print(end-start)
    # #
    # #
    update_temperature_from_kma_API_to_csv()

if __name__=='__main__':
    file_names = ['data\\all_2.csv']

    for file in file_names:
        train(file)