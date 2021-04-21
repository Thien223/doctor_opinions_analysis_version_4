###import library
import argparse

from Applications.TextClassification.preprocessing import preprocessor
from Applications.TextClassification.model import Unilabel_Classifier
from Applications.TextClassification.utils import *
from hparams import create_hparams
import torch, numpy as np, os
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def validation(valloader, model, label_path):
	labels_text, label_class_sequences = load_labels(label_path)
	model.eval()
	count = 0
	mses=[]
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(valloader):
			pred_outputs, _ = model(inputs, debug=False)
			for j in range(pred_outputs.size(0)):
				pred_label,_,mse = get_pred_label(labels_text, np.asarray(label_class_sequences), pred_outputs[j].cpu())
				true_label,_,_ = get_pred_label(labels_text, np.asarray(label_class_sequences), labels[j].cpu())
				if pred_label == true_label:
					count += 1
				print(f'pred:\t{pred_label}\ttrue:\t{true_label}\tmse:\t{mse}')
				mses.append(float(mse))
			# if i>=int(6000/hparams.batch_size):
			# 	break
	score = (count / (batch_size * (i + 1))) * 100
	# df = pd.DataFrame(mses, columns=['pred','true','mse'])
	# df.to_excel('validation.xlsx')
	threshold = mses[int(len(mses)*0.98)]
	# threshold = max(mses)
	print(f'--- validation accuracy: {score}%')
	return score, threshold


def validation_with_softmax(valloader, model):
	model.eval()
	count = 0
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(valloader):
			pred_outputs = model(inputs)
			target = torch.argmax(labels, dim=1)
			pred = torch.argmax(pred_outputs, dim=1)
			count += torch.sum(target==pred)
			print(f'--- pred: {target} \n--- true: {pred}---')
	score = (count / (batch_size * (i + 1))) * 100
	print(f'--- validation accuracy: {score}%')
	return score


def train(args,model, optimizer, train_dataloader, val_dataloader, iteration):
	criterion = torch.nn.MSELoss()
	model.train()
	best_acc=0
	for epoch in range(epochs):
		print(f'------ Epoch: {epoch} -----')
		for inputs, labels in train_dataloader:
			model.zero_grad()
			pred_outputs, _ = model(inputs)
			loss = criterion(pred_outputs, labels)
			with open(f'logs/{args.hospital}/last_training_losses.txt','a+') as f:
				f.write(str(round(float(loss),2))+'\n')
			loss.backward()
			optimizer.step()
			# if iteration % 600 == 0:
			# 	print(f'pred_outputs -- {pred_outputs} --- labels --- {labels}')
			print(f'iter -- {iteration} --- loss --- {loss}')
			if (iteration) % validate_interval == 0:
				accuracy, _ = validation(valloader=val_dataloader, model=model, label_path=f'processed/{args.hospital}/labels.txt')
				with open(f'logs/{args.hospital}/last_validation_accuracy.txt', 'a+') as f:
					f.write(str(round(float(accuracy), 4)) + '\n')
				if accuracy > best_acc:
					best_acc=accuracy
					filepath = os.path.join(checkpoint_folder, f'inception_best.model')
					torch.save({'iteration': iteration,
					            'state_dict': model.state_dict(),
					            'optimizer': optimizer.state_dict(),
					            'learning_rate': 0.0005,'threshold': None,
				            'noise': None}, filepath)

					print(f"\n=================================================")
					print(f'Saved best checkpoint to: {filepath}')
					print(f"=================================================\n")


				filepath = os.path.join(checkpoint_folder, f'{iteration}_acc_{accuracy}.model')
				torch.save({'iteration': iteration,
				            'state_dict': model.state_dict(),
				            'optimizer': optimizer.state_dict(),
				            'learning_rate': 0.0005,
				            'threshold': None,
				            'noise': None}, filepath)
				print(f"\n=================================================")
				print(f'Saved checkpoint to: {filepath}')
				print(f"=================================================\n")

			iteration += 1
	return model


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--hospital', type=str,default='hospital_01',help='hostpital id to train')
	parser.add_argument('--from_checkpoint', default=False, action='store_true',help='whether training from pretrained checkpoint')
	parser.add_argument('--only_tuning', default=False, action='store_true',help='whether do the training or just tuning')
	parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint to restore from')
	args = parser.parse_args()
	return args

if __name__=='__main__':
	args = get_args()
	hparams = create_hparams()
	### preprocessing data: and save data to "preprocessed/train.csv run once only
	# df = convert_code_to_name(dease_name_code_path=r'dataset/3차 데이터/질환코드.csv', processed_data_path='processed/total_processed.csv', encoding='cp949')
	# training params
	epochs = 5000
	batch_size = hparams.batch_size
	validate_interval = 3000  ## validation 데이터 실행 주기
	### load tokenizer
	import pickle, random


	os.makedirs(f'processed/{args.hospital}', exist_ok=True)
	os.makedirs(f'logs/{args.hospital}', exist_ok=True)
	tokenizer_path=f'processed/{args.hospital}/tokenizer.pickle'
	label_path=f'processed/{args.hospital}/labels.txt'




	# print(f"\n=================================================")
	# print(f"Preprocess data for params turning..")
	# print(f"=================================================\n")
	# tune_input_sequences, tune_labels_sequences, _, _ = preprocessor.process(doctor_opinions_df=all_data,
	#                                                                          tokenizer=tokenizer,
	#                                                                          label_path=f'processed/{args.hospital}/labels.txt',
	#                                                                          balance=True)
	#
	# ## random shuffling
	# zipped = list(zip(tune_input_sequences, tune_labels_sequences))
	# random.shuffle(zipped)
	# tune_input_sequences, tune_labels_sequences = zip(*zipped)
	# tune_dataloader = data_generator(tune_input_sequences[int(len(tune_input_sequences)*0.8):], tune_labels_sequences[int(len(tune_labels_sequences)*0.8):], batch_size=batch_size)

	# ood_dataloader = data_generator(ood_input_sequence,ood_labels_sequences,batch_size=hparams.batch_size)



	print(f"\n=================================================")
	print(f"Preprocess data for training..")
	print(f"=================================================\n")
	### load data and transform to sequences
	all_data = get_data_from_hospital(args.hospital)

	train_input_sequences, train_labels_sequences, ood_input_sequence, ood_labels_sequences, tokenizer = preprocessor.process(doctor_opinions_df=all_data,
	                                                                                                               tokenizer_path=tokenizer_path,
	                                                                                                               label_path=label_path,
	                                                                                                               balance=False)


	## random shuffling
	zipped = list(zip(train_input_sequences, train_labels_sequences))
	random.shuffle(zipped)
	train_input_sequences, train_labels_sequences = zip(*zipped)
	### c0r0eate data generator for training
	train_dataloader = data_generator(train_input_sequences[:int(len(train_input_sequences) * 0.8)], train_labels_sequences[:int(len(train_input_sequences) * 0.8)], batch_size=batch_size)
	val_dataloader = data_generator(train_input_sequences[int(len(train_input_sequences) * 0.8):], train_labels_sequences[int(len(train_input_sequences) * 0.8):], batch_size=batch_size)

	checkpoint_folder = f'checkpoints/{args.hospital}'
	os.makedirs(checkpoint_folder, exist_ok=True)

	#### load model
	if args.from_checkpoint:
		assert args.checkpoint_path is not None, 'if you want to restore checkpoint, please pass a checkpoint path..\nnormally, checkpoint path saved in checkpoints/<..>_best.model..'
		checkpoint_path = args.checkpoint_path
	else:
		checkpoint_path = None

	model = Unilabel_Classifier(words_count=len(tokenizer.word_counts), hparams=hparams).to(device)
	if checkpoint_path is None:
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
		iteration = 1
	else:
		model, optimizer, learning_rate, iteration, _, _ = load_checkpoint(checkpoint_path, model)


	os.makedirs('processed', exist_ok=True)
	with open('processed/val_dataloader', 'wb') as handle:
		pickle.dump(val_dataloader, handle, protocol=pickle.HIGHEST_PROTOCOL)
	if not args.only_tuning:
		model = train(args, model, optimizer, train_dataloader=train_dataloader, val_dataloader=val_dataloader, iteration=iteration)


	### making sure we use the best model
	model, optimizer, learning_rate, iteration, _,_ = load_checkpoint(os.path.join(checkpoint_folder, 'inception_best.model'), model)
	accuracy, threshold = validation(valloader=val_dataloader, model=model, label_path=f'processed/{args.hospital}/labels.txt')
	# best_params_idx, noise, threshold = params_turning(hsptcd=args.hospital, model=model)

	print(f"\n=================================================")
	print(f"best threshold: {threshold}")
	print(f"best model's validation accuracy: {accuracy}")
	print(f"=================================================\n")

	torch.save({'state_dict': model.state_dict(),
	            'optimizer': optimizer.state_dict(),
	            'iteration': iteration,
	            'learning_rate': 0.0005,
	            'threshold': threshold,
	            'noise': None}, os.path.join(checkpoint_folder, 'inception_best.model'))
	print(f"\n=================================================")
	print(f'Training Finished. Saved model to: {os.path.join(checkpoint_folder, "inception_best.model")}')
	print(f"=================================================\n")


# 
# 
# 
# from urllib.request import Request, urlopen
# req = Request('https://miro.medium.com/max/4800/1*MS_sC3rpdyOGSJF8rwoJxA.png', headers={'User-Agent': 'Mozilla/5.0'})
# output = open("file01.jpg","wb")
# output.write(urlopen(req).read())
# output.close()
# 
# 
# 
