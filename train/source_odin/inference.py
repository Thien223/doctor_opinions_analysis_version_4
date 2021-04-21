
from hparams import create_hparams
from Applications.TextClassification.utils import *
hparams = create_hparams()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
### load label texts
label_path = 'processed/labels.txt'
labels_text, label_class_sequences = load_labels(label_path)
################################# 	  load tokenizer 	#################################
with open('processed/tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)

def load_model(hsptcd):
	from Applications.TextClassification.model import Unilabel_Classifier
	################################# 	     Models   	 	#################################
	hparams = create_hparams()
	model = Unilabel_Classifier(words_count=len(tokenizer.word_counts), hparams=hparams)
	model.eval().to(device)


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



# def inference(opinions_sequences, model=None, tokenizer = None ,noise=-0.0098, threshold=6.6): 74.48%
# def inference(opinions_sequences, model=None, tokenizer = None ,noise=-0.0112, threshold=4.2): 76.47%
# def inference(opinions_sequences, model=None, tokenizer = None ,noise=-0.0112, threshold=3.85): 0.7540531880807434
# def inference(opinions_sequences, model=None, tokenizer = None ,noise=-0.0112, threshold=4.05): 76.2%
# def inference(opinions_sequences, model=None, tokenizer = None ,noise=-0.0112, threshold=4.15): 0.7548542133931432
def inference(opinions_sequences, model=None, tokenizer = None ,noise=-0.0112, threshold=4.2):
	### get hyper params
	### load tokenizer
	if tokenizer is None:
		with open('processed/tokenizer.pickle', 'rb') as handle:
			tokenizer = pickle.load(handle)

	if model is None:
		### load pretrainded model
		checkpoint_path = 'checkpoints/315999_acc_93.3344414893617.model'
		from Applications.TextClassification.model import Unilabel_Classifier
		model = Unilabel_Classifier(words_count=len(tokenizer.word_counts), hparams=hparams)
		model, _, _, _ = load_checkpoint(checkpoint_path, model)

	### create MSE loss instance to calculate the loss

	##set model mode to eval
	model.eval()
	opinions_sequences = torch.from_numpy(opinions_sequences).float()
	# with torch.no_grad():
	### predict the output with original input
	pred_outputs, embedded = model(opinions_sequences)
	### get label and calculate the loss
	pred_label,pred_sequence, mse_ = get_pred_label(labels_text, np.asarray(label_class_sequences), pred_outputs[0])
	criterion = torch.nn.MSELoss()
	loss = criterion(pred_outputs, pred_sequence.float().unsqueeze(dim=0))
	loss.backward()
	### extract the gradient
	embed_gradient = torch.ge(embedded.data,0)
	embed_gradient = (embed_gradient.float() - 0.5 ) * 2
	embed_gradient = noise * embed_gradient
	### predict label again with modified gradient
	new_pred, _ = model(opinions_sequences, embed_gradient=embed_gradient)

	### get new predicted label, and mse score
	new_pred_label,_, new_mse = get_pred_label(labels_text, np.asarray(label_class_sequences), new_pred)
	# ### compare score to threshold
	if new_mse >= threshold:
		label='unknown'
	else:
		label = pred_label
	return label, mse_, new_mse, model


def test_performance(model):
	with open('processed/tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	model = None
	ood_filepath = f'processed/out_of_distribution_classes.csv'
	iod_filepath = f'processed/in_of_distribution_classes.csv'
	## IOD: in of distribution (classes that exist in training data)
	## OOD: out of distribution (classes that do not exist in training data)
	import pandas as pd
	ood_df = pd.read_csv(ood_filepath, encoding='cp949')
	ood_df.drop_duplicates(keep="first", inplace=True)
	ood_df['DSES_NM'] = 'unknown'

	iod_df = pd.read_csv(iod_filepath, encoding='cp949')
	iod_df.drop_duplicates(keep="first", inplace=True)
	total_df = pd.concat((ood_df,iod_df))
	total_df = total_df.sample(frac=1)
	# iod_df = iod_df.sample(frac=1)
	result = []
	count=0
	for row in total_df[['소견','DSES_NM']][:5000].itertuples():
		opinions = row[1]
		label = row[2]

		### preprocessing the input text
		cleaned_opinions = [hangul_preprocessing(doctor_opinions=opinions.replace('\n', ' ').strip())]
		cleaned_label = [hangul_preprocessing(doctor_opinions=label.replace('\n', ' ').strip())]
		opinions_sequences, label_sequence, tokenizer = words_transform(opinionslist=cleaned_opinions,
		                                                   labelslist=cleaned_label,
		                                                   input_len=hparams.input_sequence_length,
		                                                   output_len=hparams.output_sequence_length,
		                                                   tokenizer=tokenizer)
		if label != 'unknown':
			label_sequence = torch.from_numpy(label_sequence).float()
			label,_,_ = get_pred_label(labels_text=labels_text,label_class_sequences=np.asarray(label_class_sequences),pred_label=label_sequence)
		pred_label, mse, new_mse, model = inference(opinions_sequences=opinions_sequences, model=model, tokenizer=tokenizer)
		if pred_label == label:
			count+=1
		result.append([opinions, pred_label, label, float(mse), float(new_mse)])
		print(f'{pred_label}\t---\t{label}\t---\t{mse}')
	export = pd.DataFrame(result,columns=['opinion', 'pred', 'real', 'mse', 'new_mse'])
	export.to_excel(f"result_{int(count)/len(export)}.xlsx")
	print(f'Accuracy: {int(count)/len(export)}')



if __name__=='__main__':
	serve(app, host='0.0.0.0', port=1111)