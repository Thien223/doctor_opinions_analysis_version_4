import argparse
import datetime
import json
import time

import keras.backend as K
import tensorflow as tf
from keras.models import load_model
from sklearn.externals import joblib

from config import hparams as config
from utils import *


def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, default='', help='Input data')
	parser.add_argument("--building_id", type=str, default='', help="Building ID to predict")
	parser.add_argument("--plot", type=bool, default=False, help="Whether plot result or not")
	parser.add_argument("--predicting_steps", type=int, default=192, help="Steps to forecast")
	args = parser.parse_args()
	return args


def predict(args):
	start = time.time()
	##get data from parameters
	data_ = args.data
	### convert to dataframe
	data_ = json.loads(data_.replace("\"", "").replace("\'", '"'))
	data_ = pd.DataFrame.from_dict(data_, orient='index', columns=['Power'])
	data_.index = pd.to_datetime(data_.index)
	###add temperature and weekday_index to dataframe
	### shift index to next 2 days
	old_index = data_.index
	new_index = old_index + datetime.timedelta(minutes=15 * config.time_steps)
	temp_df = data_.set_index(new_index)
	## add temperature and weekday_index (of next 2 days)
	new_data = add_hourly_temp_column_from_csv(temp_df)
	data_ = add_weekday_column(new_data)
	##set index back to original one
	data = data_.set_index(old_index)
	##get modelname (or building_id)
	model_name = args.building_id
	scaler_path = 'pretrained_models/linear_activation/{}.scaler'.format(model_name)
	try:
		### load scaler from saved file
		scaler = joblib.load(scaler_path)
	except KeyError as e:
		raise KeyError('dumping and loading libraries are not matched. Check joblib version..')
	### get predicting time steps from parameters
	try:
		predicting_steps = int(args.predicting_steps)
	except ValueError as e:
		raise ValueError('Predicting steps must be an integer..')
	### past temperature
	temp = pd.read_csv('temperature.csv')
	temp[temp.columns[0]] = pd.to_datetime(temp[temp.columns[0]])
	temp.set_index(temp.columns[0], inplace=True)
	## transform to 15 minutes frequency
	temp = temp.resample('15T').interpolate()
	### define forecast length
	forecastLength = predicting_steps
	forecasted = []
	#### initialize input window (used only last part of data to save inferencing time)
	window = data[-config.time_steps - 1:]
	# window = train_data
	next_index = list(window.index)
	window = scaler.transform(window)
	#### sequentializing input data
	input_, output_ = sequentialize_multiple_features_data(scaled_inputData=window, inputData_index=next_index, output_indexes=[0], config=config)
	with tf.Session(graph=K.get_session().graph) as session:
		session.run(tf.global_variables_initializer())
		### inference
		model = load_model('pretrained_models/linear_activation/{}.h5'.format(model_name))
		for i in range(forecastLength):
			## add next time index to old index list
			next_index.append(next_index[-1] + datetime.timedelta(minutes=15))
			### define index of temperature and weekday (we take temperature and weekday of next 7days) 15*config.time_steps = 7days
			next_temp_and_weekday_index = next_index[-1] + datetime.timedelta(minutes=15 * config.time_steps)
			## predict
			predictedSequence = model.predict(input_)
			## add result to result list
			forecasted.append(float(predictedSequence[-1]))
			#### preparing next prediction input####
			########################################
			## get temperature
			temperature_ = temp['Temperature'][str(next_temp_and_weekday_index)]
			## get weekday index
			weekday_index_ = get_weekday_index(next_temp_and_weekday_index)
			### add them to new row (we add 0 to array to make sure array has 3 columns)
			next_row_ = np.array([[0, temperature_, weekday_index_]])
			## transform temperature and weekday index scale (the scaler require input has 3 columns)
			next_row_ = scaler.transform(next_row_)
			## replace predicted value by original one
			next_row_[0][0] = predictedSequence[-1]
			## expand dimension of row to 3D array [batch_size, time_steps, features] (batch_size=1)
			next_row = np.expand_dims(next_row_, axis=1)
			## remove the last element and add predicted element into first possition
			input_ = np.concatenate((next_row, input_[:, :-1, :]), axis=1)
	### transform predicted data to normal scale
	forecasted_np = np.array(forecasted).reshape(-1, 1)
	### add 2 columns (since scaler require 3 columns shape)
	forecasted_np_ = np.concatenate((forecasted_np, np.zeros(shape=(len(forecasted), 2))), axis=1)
	## rescale to normal energy scale
	forecasted_np_rescaled = scaler.inverse_transform(forecasted_np_)
	## add time index (since result has 3 columns, we just take the output column at the first possition)
	forecasted_df = pd.DataFrame(forecasted_np_rescaled[:, 0], index=next_index[-forecastLength:])
	end = time.time()
	print('Total time (s): {}'.format(end - start))
	if args.plot == True:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.style as style
		style.use('fivethirtyeight')
		import matplotlib.pyplot as plt
		### plot
		plt.ioff()
		plt.plot(forecasted_df, label='Predicted')
		plt.plot(data['Power'], label='Real seen')
		plt.legend()
		plt.grid()
		plt.savefig('result.png')
	return forecasted_df


if __name__ == '__main__':
	args = get_arguments()
	predicted = predict(args)
	for row in predicted.itertuples():
		print(dict({'Time': str(row[0]), 'Power': str(row[1])}))
	print('Predicting was done, updating temperature data..')
	update_temperature_from_kma_API_to_csv()
