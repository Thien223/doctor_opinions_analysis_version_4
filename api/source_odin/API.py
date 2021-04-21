import pickle
import os
import numpy as np
import random
import torch
import argparse
from flask import Flask, render_template, request, json
from waitress import serve
from Applications.TextClassification.utils import *
from hparams import create_hparams
from Applications.TextClassification.model import Unilabel_Classifier
from Applications.TextClassification.preprocessing import preprocessor

torch.manual_seed(1542)
torch.cuda.manual_seed(1542)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1542)
np.random.seed(1542)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hparams = create_hparams()
## set current working directory to source code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("\nCurrent working directory: ---'{}'---\n".format(os.getcwd()))

app = Flask(__name__)



def load_model(args):
	#checkpoint_path = f'../checkpoints/{hsptcd}/inception_best.model'
	#tokenizer_path = f'../processed/{hsptcd}/tokenizer.pickle'
	#label_path = f'../processed/{hsptcd}/labels.txt'
	checkpoint_path = args.checkpoints
	tokenizer_path = os.path.join(args.processed,'tokenizer.pickle')
	label_path = os.path.join(args.processed,'labels.txt')
	### load label texts
	labels_text, label_class_sequences = load_labels(label_path)


	with open(tokenizer_path, 'rb') as handle:
		tokenizer = pickle.load(handle)
	### load pretrainded model
	model = Unilabel_Classifier(words_count=len(tokenizer.word_counts), hparams=hparams)
	model, optimizer,_,_, threshold, noise = load_checkpoint(checkpoint_path, model)
	model.eval().to(device)
	print(f"\n============== SUMARY ===========================")
	print(f"Loaded model from: {checkpoint_path}")
	print(f"Loaded tokenizer from: {tokenizer_path}")
	print(f"Loaded label from: {label_path}")
	print(f"threshold: {threshold}")
	print(f"=================================================\n")
	return model, tokenizer_path, labels_text, label_class_sequences, threshold, noise


@app.route('/')
def index():
	return render_template('index.html')


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoints', type=str,default='../checkpoints/inception_best.model',help='checkpoint to load')
	parser.add_argument('--processed', type=str,default='../processed/',help='labels.txt and tokenizer.pickle files holding folder')
	parser.add_argument('--delimiter', type=str, default='|', help='delemiter of each opinions')
	args = parser.parse_args()
	return args

args = get_args()
model, tokenizer_path, labels_text, label_class_sequences, threshold, noise = load_model(args)

def inference(opinions, hsptcd=None):
	# model, tokenizer, labels_text, label_class_sequences, threshold, noise = load_model(hsptcd=hsptcd)
	model.zero_grad()
	print(f'=========== input opinion: =====================')
	print(f'{opinions}')
	print(f'================================================\n')

	opinions = opinions.split(args.delimiter)

	## preprocessing the input text
	cleaned_opinions = [hangul_preprocessing(doctor_opinions=opi.replace('\n', ' ').strip()) for opi in opinions]
	print(f'cleanned opinions: {cleaned_opinions}')
	opinions_sequences, _, _ = words_transform(opinionslist=cleaned_opinions,
	                                           labelslist=None,
	                                           input_len=hparams.input_sequence_length,
	                                           output_len=hparams.output_sequence_length,
	                                           tokenizer_path=tokenizer_path)
	opinions_sequences = torch.from_numpy(opinions_sequences).float()
	opinions_sequences = opinions_sequences.to(device)
	print(f'=========== opinions sequences: ================')
	print(f'{opinions_sequences}')
	print(f'================================================\n')
	with torch.no_grad():
	### predict the output with original input
		pred_outputs, embedded = model(opinions_sequences)

	### get label and calculate the loss
	labels = []
	mses = []
	for i in range(pred_outputs.size(0)):
		pred_label, pred_sequence, mse_ = get_pred_label(labels_text, np.asarray(label_class_sequences), pred_outputs[i])
		if mse_ >= threshold:
			# label = pred_label
			pred_label = 'unknown'
		labels.append(pred_label)
		mses.append(mse_)
	print(f'=========== mses: ================')
	print(f'{mses}')
	print(f'=================================\n')
	# criterion = torch.nn.MSELoss()
	# loss = criterion(pred_outputs, pred_sequence.float().unsqueeze(dim=0).to(device))
	# loss.backward()
	# ### extract the gradient
	# embed_gradient = torch.ge(embedded.data, 0)
	# embed_gradient = (embed_gradient.float() - 0.5) * 2
	# embed_gradient = noise * embed_gradient
	# ### predict label again with modified gradient
	# new_pred, _ = model(opinions_sequences, embed_gradient=embed_gradient)
	# ### get new predicted label, and mse score
	# new_pred_label, _, new_mse = get_pred_label(labels_text, np.asarray(label_class_sequences), new_pred)
	# ### compare score to threshold

	torch.cuda.empty_cache()

	return labels, mses



def load_test_data(path ='../processed/in_of_distribution_classes.csv'):
	import pandas as pd
	df = pd.read_csv(path, encoding='cp949')
	df = df[['소견','DSES_CD','DSES_NM']].sample(frac=1)
	return df

@app.route('/classify', methods=['GET','POST'])
def classify():
	try:
		request_data = request.get_json()
		svckey = request_data['svckey']  ### service key Af5HQBAvkv
		if str(svckey) == 'Af5HQBAvkv':
			hsptcd = request_data['hsptcd']  ### hostpital code
			cnts = request_data['cnts']  ### opinions
			pid = request_data['pid']  ### patient id
			labels,_ = inference(opinions=cnts, hsptcd=hsptcd)
			data = {'hsptcd': hsptcd, 'pid': pid, 'oicn': labels}
			res = {'result': 'ok', 'msg': 'successed', 'cnt': len(labels), 'data': data}
			return json.dumps(res, ensure_ascii=False)
		else:
			res = {'result': 'error', 'msg': 'Unusable service key (svckey)', 'cnt': 0, 'data': {}}
			return json.dumps(res, ensure_ascii=False)
	except TypeError as e:
		request_data = request.args
		svckey = request_data.get('svckey')  ### service key Af5HQBAvkv
		if str(svckey) == 'Af5HQBAvkv':
			hsptcd = request_data.get('hsptcd')  ### hostpital code
			cnts = request_data.get('cnts')  ### opinions
			pid = request_data.get('pid')  ### patient id
			labels, _ = inference(opinions=cnts, hsptcd=hsptcd)
			data = {'hsptcd': hsptcd, 'pid': pid, 'oicn': labels}
			res = {'result': 'ok', 'msg': 'successed', 'cnt': len(labels), 'data': data}
			return json.dumps(res, ensure_ascii=False)
		else:
			res = {'result': 'error', 'msg': 'Unusable service key (svckey)', 'cnt': 0, 'data': {}}
			return json.dumps(res, ensure_ascii=False)
	except Exception as e:
		err_msg = e
		res = {'result': 'error', 'msg': err_msg, 'cnt': 0, 'data': {}}
		return json.dumps(res, ensure_ascii=False)


if __name__=='__main__':

	serve(app, host='0.0.0.0', port=3023)