import json
import os, numpy as np
import pickle
import re
import torch
import psycopg2
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
import pandas as pd



def data_generator(input_sequences, labels_sequences, batch_size=8):
	from torch.utils.data import TensorDataset, DataLoader
	input_sequence = torch.tensor(input_sequences, dtype=torch.long, device=device)
	output_sequence = torch.tensor(labels_sequences, dtype=torch.float, device=device)
	dataset = TensorDataset(input_sequence, output_sequence)
	dataloader = DataLoader(dataset, batch_size=batch_size)
	return dataloader

def hangul_preprocessing(doctor_opinions, remove_stopwords=True, stopwords=None):
	'''
	process hangul to list of words
	:param doctor_opinions: documents to analysis
	:param remove_stopwords: whether remove the stopwords or not
	:param stopwords: list of stopwords to remove
	:return: list of words in document
	'''
	doctor_opinions_cleaned=None
	if str(doctor_opinions)!='nan':
		stopwords = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한']
		from konlpy.tag import Okt
		## remove all non-hangul and space chars

		doctor_opinions_cleaned = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣa-z\\s]", "", doctor_opinions)
		while True:
			if '  ' in doctor_opinions_cleaned:
				doctor_opinions_cleaned=doctor_opinions_cleaned.replace('  ', ' ')
			else:
				break
		open_korean_text = Okt()
		## extract words
		doctor_opinions_cleaned = open_korean_text.morphs(doctor_opinions_cleaned, stem=True)
		### remove stopwords in words list
		if remove_stopwords:
			assert stopwords is not None, 'when remove_stopwords is True, require stopwords list'
			doctor_opinions_cleaned = [token for token in doctor_opinions_cleaned if not token in stopwords]
	return doctor_opinions_cleaned

def pad_list(list,max_leng,pad_val=0):
	new_list = [pad_val for _ in range(max_leng)]
	new_list[:len(list)] = list
	return new_list


def words_transform(opinionslist, labelslist, input_len, output_len,tokenizer_path=None):
	'''
	transform words in  to sequences
	:param wordslist: list of all processed opinions
	:return: sequences lists of each opinions
	'''
	from tensorflow.python.keras.preprocessing.sequence import pad_sequences
	from tensorflow.keras.preprocessing.text import Tokenizer
	max_input_sequence_length = input_len
	max_output_sequence_length = output_len
	if labelslist is not None:
		wordslist = opinionslist + labelslist
	else:
		wordslist=opinionslist
	save_tokenizer = False


	#### if tokenier path is not a file, create it
	if not os.path.isfile(tokenizer_path):
		save_tokenizer = True
		tokenizer = Tokenizer()
		### train tokenizer
		tokenizer.fit_on_texts(wordslist)
	else: ### if tokenizer is exist, load it
		with open(tokenizer_path, 'rb') as handle:
			tokenizer = pickle.load(handle)
	if save_tokenizer:
		with open(tokenizer_path, 'wb') as handle:
			pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

	opinions_sequences = tokenizer.texts_to_sequences(opinionslist)
	opinions_sequences = pad_sequences(opinions_sequences, maxlen=max_input_sequence_length, padding='post')
	if labelslist is not None:
		labels_sequences = tokenizer.texts_to_sequences(labelslist)
		labels_sequences = pad_sequences(labels_sequences, maxlen=max_output_sequence_length, padding='post')
		return opinions_sequences, labels_sequences, tokenizer
	else:
		return opinions_sequences, None, tokenizer


def get_pred_label(labels_text, label_class_sequences, pred_label):
	from torch.nn import functional as F
	import numpy as np
	pred_label = pred_label.cpu()
	## convert to torch tensor to use with F.mse_loss
	label_class_sequences = torch.from_numpy(label_class_sequences)
	mse_losses = []
	### find the loss of pred sequence with each class sequence
	for i, label in enumerate(label_class_sequences):
		mse_loss = F.mse_loss(label, pred_label.squeeze())
		mse_loss = mse_loss / torch.mean(pred_label)
		mse_losses.append(mse_loss)
	### take the minimum mse loss as predicted sequence
	matched_sequence = label_class_sequences[mse_losses.index(min(mse_losses))]
	### take the equivalent labels text as predicted label
	matched_label = labels_text[int(np.where(np.all(label_class_sequences.data.cpu().numpy() == matched_sequence.data.cpu().numpy(), axis=1))[0])]
	return matched_label, matched_sequence, min(mse_losses)

def load_checkpoint(checkpoint_path, model):
	assert os.path.isfile(checkpoint_path)
	checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device(device))
	model.load_state_dict(checkpoint_dict['state_dict'])
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
	optimizer.load_state_dict(checkpoint_dict['optimizer'])
	iteration = checkpoint_dict['iteration']
	learning_rate = checkpoint_dict['learning_rate']
	try:
		threshold = checkpoint_dict['threshold']
		noise = checkpoint_dict['noise']
	except Exception as e:
		threshold, noise = None, None
	return model, optimizer, learning_rate, iteration, threshold, noise




def load_labels(label_path='processed/labels.txt'):
	f = open(label_path, 'r')
	labels= f.read().replace('\'','\"')
	labels=json.loads(labels)
	f.close()
	label_texts = []
	label_sequence=[]
	for k,v in labels.items():
		label_texts.append(k)
		label_sequence.append(v)
	return label_texts, label_sequence


def load_tokenizer(tokenizer_path):
	assert os.path.isfile(tokenizer_path), "model checkpoint must be a file"
	with open(tokenizer_path, 'rb') as handle:
		tokenizer = pickle.load(handle)
	return tokenizer


def scale_to_0and1(data):
	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler(copy=False,feature_range=(0, 1))  # MinMax Scaler
	data = scaler.fit_transform(data)  # input: ndarray type data
	return (data, scaler)



def load_data(data_path, encoding='utf-8'):
	import pandas as pd
	if data_path.endswith('.csv'):
		df = pd.read_csv(data_path, encoding=encoding)
	elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
		df = pd.read_excel(data_path, encoding=encoding)
	else:
		raise NotImplementedError("Not supported file type, supporting [.xls, .xlsx, .csv]..")
	return df


def process_data(datapath):
	'''
	process data table [id,BNCH_CD, ORD_SEQ, ORD_CD, ORD_NM_INFO, CHK_CMT_OPN, DSES_CD, CHOS_NO] to 소견-질환코드
	first: join all 소견 that have same ORD_CD,
	second: get the corresponding DSES_CD
	third: export to csv file

	:param datapath: path to data
	:return:
	'''
	data_prefix = datapath.split('\\')[-1].split('.')[0]
	splitted_opinions_df = load_data(datapath)
	orders = list(set(splitted_opinions_df['ORD_SEQ']))
	joined_opinion = []
	joined_dease_code = []
	for k in orders:
		opinion = None
		dease_cd=None
		for row in splitted_opinions_df.loc[splitted_opinions_df['ORD_SEQ']==k].itertuples():
			opinion = row[6] if opinion is None else opinion+'\n'+row[6]
			if str(row[7]).strip() != 'nan':
				dease_cd = row[7] if dease_cd is None else dease_cd+'\n'+ str(row[7])
		if (opinion is not None) and (dease_cd is not None):
			joined_opinion.append(opinion)
			joined_dease_code.append(dease_cd.strip())

	import pandas as pd
	df = pd.DataFrame({'소견':joined_opinion,'DSES_CD':joined_dease_code})
	df.to_csv(f'processed_{data_prefix}.csv',encoding='cp949')
	print(f'exported to: processed_{data_prefix}.csv')

def expand_all(df, cols, seps):
	import pandas as pd
	def expand(df, col, sep=','):
		r = df[col].str.split(sep)
		d = {c: df[c].values.repeat(r.str.len(), axis=0) for c in df.columns}
		d[col] = [i for sub in r for i in sub]
		return pd.DataFrame(d)

	ret = df
	for c,s in zip(cols,seps): ret = expand(ret,c,s)
	return ret



def convert_code_to_name(dease_name_code_path, processed_data_path,encoding='utf-8'):
	dease_name_df = load_data(dease_name_code_path, encoding=encoding)
	df = load_data(processed_data_path)
	df = df.drop(columns=[df.columns[0],df.columns[1]],axis=1)
	# df = df.dropna()
	# df = expand_all(df=df,cols=['DSES_CD'],seps=[','])
	non_label_df = df.loc[df['DSES_CD'].isnull()]
	df['DSES_NM'] = df['DSES_CD'].map(dease_name_df.set_index('DSES_CD')['DSES_NM'])

	non_label_df.to_csv('processed\\non-label-opinions.csv', encoding='cp949')
	print(f'number of data rows before removing multi labels data: {len(df)}')
	print(f'number of classes before filtering dease name: {len(set(df["DSES_NM"]))}')
	df = df.dropna() ### remove multi label rows
	print(f'number of data rows  before filtering dease name and after removing multi labels data: {len(df)}')
	print(f'number of classes after filtering dease name and removing multi labels data: {len(set(df["DSES_CD"]))}')
	# print(f'number of classes after filtering dease name and removing multi labels data: {len(set(df["소견"]))}')
	import pandas as pd
	# dease_names = list(set(df["DSES_NM"]))
	# for name in dease_names:
	# 	print(f'질환명: {name} -- 개수: {len(df.loc[df["DSES_NM"]==name])}')
	df = df.drop_duplicates()
	df[['소견','DSES_CD','DSES_NM']].to_csv('train.csv',encoding='cp949')
	# df.describe()
	print(f'Exported data to: "train.csv"')
	return df


def join_data(encoding='utf-8'):
	import pandas as pd
	import os
	df = None
	for file in os.listdir('./'):
		if 'processed_20200' in file and file.endswith('.csv'):
			df = load_data(file) if df is None else pd.concat((df, load_data(file)), axis=0)
	df.to_csv('total_processed.csv', encoding=encoding)
	return df




def get_data_from_hospital(hospital='hospital_01'):
	from dotenv import load_dotenv
	# env path
	load_dotenv(dotenv_path='source_odin/.env')
	try:
		conn = psycopg2.connect(host=os.getenv('host'), port=os.getenv('port'), user=os.getenv('user'), password=os.getenv('password'), database=os.getenv('database'))
	except ConnectionError as e:
		raise ConnectionError('Error when connecting to database server, check the network connection or login information./.')
	cur = conn.cursor()
	query = f"select {hospital}_train.id, {hospital}_train.opinion, {hospital}_train.dses_cd, {hospital}_train.dses_nm from {hospital}_train inner join {hospital}_label on {hospital}_train.dses_nm={hospital}_label.dses_nm;;"
	try:
		cur.execute(query)
	except Exception:
		raise Exception(f'Could not find the table {hospital}, please put data to the corresponding table before running the training..')
	fetched = cur.fetchall()
	data = pd.DataFrame(fetched, columns=['id', '소견', 'DSES_CD', 'DSES_NM'])
	data.set_index('id',inplace=True)
	cur.close()
	conn.close()
	return data

def OOD_params_tuning(model, iod_dataloader, ood_dataloader, hsptcd):
	from hparams import create_hparams
	### load pretrainded model
	criterion = torch.nn.MSELoss()
	save_path = f'processed/{hsptcd}/mse_score'
	os.makedirs(save_path, exist_ok=True)
	processed_path = 'processed'
	label_path = os.path.join(processed_path, hsptcd,'labels.txt')
	labels_text, label_class_sequences = load_labels(label_path)
	model.eval().to(device)

	noise = 0  ### start with noise=-0.014 and end with noise = 0.014
	gap = 0.114 / 20  ### noise's increment steps
	while True:
		noise = noise + gap
		if noise >= 0.114:
			break
		print(f'noise: {noise}')
		count = 0
		i = 0
		iod_mse = []
		ood_mse = []
		for i, (iod_batch, ood_batch) in enumerate(zip(iod_dataloader,ood_dataloader)):
			model.zero_grad()
			# print(i)
			iod_input,iod_label = iod_batch
			pred_outputs, embed_gradient = model(iod_input, debug=False)
			loss = criterion(pred_outputs, iod_label)
			loss.backward()
			### get embedded gradient matrix
			embed_grad = torch.ge(embed_gradient.data, 0)
			### convert gradient to [-1,1] corresponding to negative and positive value of gradient
			embed_grad = (embed_grad.float() - 0.5) * 2
			### add noise to gradient
			embed_grad = noise * embed_grad
			### predict new output with new gradient
			new_pred, _ = model(iod_input, embed_gradient=embed_grad)

			for j in range(pred_outputs.size(0)):
				pred_label,_, mse = get_pred_label(labels_text=labels_text, label_class_sequences=np.asarray(label_class_sequences), pred_label=pred_outputs[j])
				true_label,_,_ = get_pred_label(labels_text, np.asarray(label_class_sequences), iod_label[j])
				if pred_label == true_label:
					count+=1
				iod_mse.append(float(mse))
			# print(f'count {count}')
		with open(os.path.join(save_path, f'modified_iod_{noise}.txt'), 'w') as f:
			f.write(("{}\n".format(iod_mse)))
		print(f'--- validation accuracy: {(100 * count / (pred_outputs.size(0) *(i + 1)))}%')

		for i, (iod_batch, ood_batch) in enumerate(zip(iod_dataloader,ood_dataloader)):
			model.zero_grad()
			ood_input,ood_label = ood_batch
			pred_outputs, embeded = model(ood_input, debug=False)
			loss = criterion(pred_outputs, ood_label)
			loss.backward()
			### get embedded gradient matrix
			embed_grad = torch.ge(embeded.data, 0)
			### convert gradient to [-1,1] corresponding to negative and positive value of gradient
			embed_grad = (embed_grad.float() - 0.5) * 2
			### add noise to gradient
			embed_grad = noise * embed_grad
			### predict new output with new gradient
			new_pred, _ = model(ood_input, embed_gradient=embed_grad)

			for j in range(pred_outputs.size(0)):
				pred_label,_, mse = get_pred_label(labels_text, np.asarray(label_class_sequences), pred_outputs[j])
				true_label,_,_ = get_pred_label(labels_text, np.asarray(label_class_sequences), ood_label[j])
				if pred_label == true_label:
					count+=1
				ood_mse.append(float(mse))
		with open(os.path.join(save_path, f'modified_ood_{noise}.txt'), 'w') as f:
			f.write(("{}\n".format(ood_mse)))






def extract_threshold(path = r'processed/mse_score'):
	'''
	after save mse score into "path", load file and extract noise and threshold (the most appropriate mse score)
	'''
	import os
	iod_files = []
	ood_files=[]
	### read files in folder and add to lists
	for file in os.listdir(path):
		file = os.path.join(path, file)
		if file.endswith('.txt'):
			if 'iod' in file:
				iod_files.append(file)
			elif 'ood' in file:
				ood_files.append(file)
			else:
				continue
	### sort file ascending
	ood_at_95_tpr_list = []
	for iod_file,ood_file in zip(iod_files,ood_files):
		with open(iod_file, 'r') as iod:
			iod_data = sorted(eval(iod.readline()))
		with open(ood_file, 'r') as ood:
			ood_data = sorted(eval(ood.readline()))

		### then find the index of the mse score that corresponding to 95% of true possitive rate
		iod_95_tpr = iod_data[int(len(iod_data) * 0.95)]
		print(f'file: {iod_file}')
		print(f'iod_95_threshold:  {int(len(iod_data) * 0.95)} --- {iod_95_tpr}')
		print(f'ood_95_threshold: {min(enumerate(ood_data), key=lambda x: abs(x[1]-iod_95_tpr))}')

		### find the nearest-to-95% tpr- in out of distribution files and add to the list
		ood_at_95_tpr_list.append(list(min(enumerate(ood_data), key=lambda x: abs(x[1]-iod_95_tpr))))

	ood_at_95_tpr_idx = np.array(ood_at_95_tpr_list, dtype=int).T
	### take the file index, best mse score, and the noise
	best_params_idx = int(np.argmin(ood_at_95_tpr_idx[0]))
	ood_idx, threshold = ood_at_95_tpr_list[best_params_idx]
	best_file = ood_files[best_params_idx]
	noise = float(best_file.split('_')[-1].replace('.txt',''))
	return best_params_idx, noise, threshold


def need_recheck(opinion):
	time_keywords = ['1일']+\
	                ['1개월', '1달', '일개월', '한달','1 개월', '1 개 월', '1개 월']+\
	                ['12개월', '12달', '십이개월', '12 개월', '12 개 월', '12개 월']+\
	                ['3개월', '3달', '삼개월', '세달','3 개월', '3 개 월', '3개 월']+\
	                ['6개월', '6달', '육개월', '여섯달','6 개월', '6 개 월', '6개 월']+\
	                ['1년', '2년']
	needed=False
	time_keywords_idx=-1
	for time in time_keywords:
		if time in opinion:
			needed = True
			time_keywords_idx = opinion.index(time)
	return needed, time_keywords_idx



def extract_recheck_opinions(doctor_opinion):
	'''
	split 의사소견 to 질환 소견
	:param doctoral_opinion: 검진소견
	:return: dictionary holds the sub opinion and the need of recheck <binary value>
	'''
	splitted_opinions = doctor_opinion.split('*')
	need_recheck_opinions = []
	for idx, opinion in enumerate(splitted_opinions):
		if len(opinion.strip()) <=1:
			continue
		needed, time_keyword_idx = need_recheck(opinion)
		if needed:
			need_recheck_opinions.append(opinion)
	return need_recheck_opinions



def params_turning(hsptcd, model):
	from Applications.TextClassification.preprocessing import preprocessor
	### load tokenizer
	import pickle, random
	with open(f'processed/{hsptcd}/tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	print(f"\n=================================================")
	print(f"Preprocess data for parameters tuning..")
	print(f"=================================================\n")
	### load data and transform to sequences
	all_data = get_data_from_hospital()
	train_input_sequences, train_labels_sequences, ood_input_sequence, ood_labels_sequences, _ = preprocessor.process(doctor_opinions_df=all_data,
	                                                                                                               tokenizer=tokenizer,
	                                                                                                               label_path=f'processed/{hsptcd}/labels.txt',
	                                                                                                               balance=True)

	## random shuffling
	zipped = list(zip(train_input_sequences, train_labels_sequences))
	random.shuffle(zipped)
	train_input_sequences, train_labels_sequences = zip(*zipped)
	### create data generator for training
	iod_dataloader = data_generator(train_input_sequences[int(len(train_input_sequences) * 0.8):], train_labels_sequences[int(len(train_input_sequences) * 0.8):], batch_size=1)
	ood_dataloader = data_generator(ood_input_sequence, ood_labels_sequences, batch_size=1)
	OOD_params_tuning(model, iod_dataloader=iod_dataloader, ood_dataloader=ood_dataloader, hsptcd=hsptcd)
	best_params_idx, noise, threshold = extract_threshold(path='processed/mse_score')

	print(f"\n=================================================")
	print(f"Finished parameters tuning..")
	print(f"=================================================\n")
	return best_params_idx, noise, threshold
