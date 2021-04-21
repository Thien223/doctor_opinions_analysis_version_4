from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
import pandas as pd
from Applications.TextClassification.utils import hangul_preprocessing, words_transform
import pickle
from hparams import create_hparams
hparams = create_hparams(None)
import os
#
# a = '''* 심장관상동맥조영CT검사결과 유소견입니다.
# [finding]
# 1. Coronary variation or anomaly: None.
# 2. CACS (Agatston score): 424.19
# 3. Atherosclerotic CAD:
# (Pls. note that the degree of stenosis suggested in this report is not quantitative but observational and empirical.)
# <Summary of Stenosis and Plaque Configurations>
# --------------------------------------------------------------------------------------------------
# Segment: Stenosis / Plaque type, shape, size /
# --------------------------------------------------------------------------------------------------
# LM,p,m-LAD: mild / calcified, mixed, nodular, tubular /
# p,m-LCX: minimal  to mild / calcified, mixed, nodular/
# p,m-RCA: minimal to mild / calcified, nodular/
# ---------------------------------------------------------------------------------------------------
# <Stenosis Degree>
# Minimal: <25%, Mild: 25-49%, Moderate: 50-69%, Severe: >70-99%, Occluded.
#
# 4. Extra-coronary CV findings:
#  - Not remarkable.
# 5. Covered lung and upper abdomen:
#  - No evidence of abnormal findings in both lungs and upper abdomen covered in this study performed by using the scanning and reconstruction protocol for CCTA
# [conclusion]'''

def process(doctor_opinions_df,tokenizer_path=None, label_path=None, balance=False):

	doctor_opinions_df.drop_duplicates(keep="first", inplace=True)
	classes = list(sorted(set(doctor_opinions_df["DSES_NM"])))

	total_ = []
	less_data = []
	for cls in classes:
		temp_df = doctor_opinions_df.loc[doctor_opinions_df["DSES_NM"]==cls]
		print(f'{cls}----{len(temp_df)}')
		if len(temp_df)>=100:
			total_.append(temp_df.sample(frac=1))
		else:
			less_data.append(temp_df.sample(frac=1))

	if balance:
		max_count = min([len(i) for i in total_])
		total_df = pd.concat([df.iloc[:max_count] for df in total_])
	else:
		total_df = pd.concat([df for df in total_])

		# less_data_df[['소견','DSES_CD','DSES_NM']].to_csv("processed/out_of_distribution_classes.csv", encoding=encoding)
	# total_df[['소견','DSES_CD','DSES_NM']].to_csv("processed/in_of_distribution_classes.csv", encoding=encoding)

	training_classes = list(sorted(set(total_df["DSES_NM"])))
	doctor_opinions_df_new = total_df
	opinions = list(doctor_opinions_df_new["소견"])
	labels = list(doctor_opinions_df_new["DSES_NM"])
	cleaned_opinions = []
	cleaned_labels = []
	for opinion, label in tqdm(zip(opinions, labels)):
		cleaned_opinion = hangul_preprocessing(doctor_opinions=str(opinion).replace('\n',' ').strip().lower())
		cleaned_label = hangul_preprocessing(doctor_opinions=str(label).strip().lower())
		cleaned_opinions.append(cleaned_opinion)
		cleaned_labels.append(cleaned_label)
	opinions_sequences, labels_sequences, tokenizer = words_transform(opinionslist=cleaned_opinions,
	                                                                  labelslist=cleaned_labels,
	                                                                  input_len=hparams.input_sequence_length,
	                                                                  output_len=hparams.output_sequence_length,
	                                                                  tokenizer_path=tokenizer_path)




	if not os.path.isfile(label_path):
		labels_text = tokenizer.sequences_to_texts(labels_sequences)
		unique_labels_text, unique_labels_sequences = [],[]
		for text, seq in zip(labels_text,labels_sequences):
			if text not in unique_labels_text:
				unique_labels_text.append(text)
				unique_labels_sequences.append(seq.tolist())
		### save labels_text and sequence for futher using
		labels_dict = dict(zip(unique_labels_text, unique_labels_sequences))
		f = open(label_path,'w')
		f.write(str(labels_dict))
		f.close()
	print('Summary:')
	print(f'input column name: {"소견"}')
	print(f'label column name: {"DSES_NM"}')
	print(f'number of labels: {len(training_classes)}')
	print(f'datarow: {len(total_df)}')

	# p22rint(f'info ----- Saved label-sequence mapping file to: {labels_sequences_path}'1)
	# 1dataloader = data_generator(opinions_sequences, labels_sequences, batch_size=hpar
	# ams.batch_size)
	### when train model using mse loss function (output is label's sequence)
	ood_opinions_sequences, ood_labels_sequences = None, None
	if len(less_data) > 0:
		less_data_df = pd.concat([df_ for df_ in less_data])
		opinions = list(less_data_df["소견"])
		labels = list(less_data_df["DSES_NM"])
		cleaned_opinions = []
		cleaned_labels = []
		for opinion, label in tqdm(zip(opinions, labels)):
			cleaned_opinion = hangul_preprocessing(doctor_opinions=str(opinion).replace('\n', ' ').strip().lower())
			cleaned_label = hangul_preprocessing(doctor_opinions=str(label).strip().lower())
			cleaned_opinions.append(cleaned_opinion)
			cleaned_labels.append(cleaned_label)
		ood_opinions_sequences, ood_labels_sequences, _ = words_transform(opinionslist=cleaned_opinions,
		                                                                  labelslist=cleaned_labels,
		                                                                  input_len=hparams.input_sequence_length,
		                                                                  output_len=hparams.output_sequence_length,
		                                                                  tokenizer_path=tokenizer_path)
	return opinions_sequences, labels_sequences, ood_opinions_sequences, ood_labels_sequences, tokenizer


	### when train model using cross-entropy loss function (output is softmax propability)
	# return opinions_sequences, labels_onehot_sequences, tokenizer.word_counts, training_classes
