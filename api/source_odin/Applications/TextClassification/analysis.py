from nltk.translate.bleu_score import sentence_bleu


def get_index_words(file='index_words.txt'):
	'''
	get 색인어 from file
	:return: list of 색인어
	'''
	index_words = []
	with open(file, 'r') as f:
		for line in f.readlines():
			index_words.append(line.strip())
	return index_words


def similarity_calculate(index_word, to_compare_word, model):
	'''
	compare similarity between 색인어 and keyword (from 소견)
	:param index_word: 저장 된 색인어
	:param to_compare_word: keyword from 소견
	:param model: fasttext model (because model loading takes time, we load it into memory and call from function)
	:return: similarity score, less is better
	'''
	import scipy.spatial.distance as distance
	index_word_vec = model.get_word_vector(index_word)
	to_compare_word_vec = model.get_word_vector(to_compare_word)
	return distance.cosine(index_word_vec, to_compare_word_vec)


def doctor_opinion_split(doctoral_opinion):
	'''
	split 의사소견 to 질환 소견
	:param doctoral_opinion: 검진소견
	:return: list of 소견 by 질환
	'''
	dease_opinions = doctoral_opinion.split('*')
	return dease_opinions[1:]


#dease_opinion=dease_opinions[1]
def dease_index_words_extract(dease_opinion):
	'''
	Get the splitted 소견 and extract the keywords
	:param dease_opinion: 질환에 대한 의사 소견
	:return: list of keywords
	'''
	from konlpy.tag import Okt
	open_korean_text = Okt()
	### get the complex nouns
	# nouns = open_korean_text.nouns(phrase='위내시경 검사결과 역류성 식도염 소견입니다. - 역류성 식도염은 흡연, 음주, 커피, 기름진 음식, 야식 등이 주된 원인입니다. 치료여부는 식도염 정도와 증상에 따라 달라지므로 내과 전문의 상담 권합니다.')
	# print(nouns)


	### get the complex nouns and nouns
	keywords = open_korean_text.phrases(phrase=dease_opinion)
	### filter out the nouns to keep only complex nouns

	# keywords = list(set(phrases) - set(nouns)) ### remove 명사, keep only 숙어
	return keywords



def extract_index_words(keywords):
	'''
	calculate similarity between filtered_phrase (keywords) and 저장된 색인어, return best matched 색인어
	:param filtered_phrases: keywords extracted from 소견
	:return: best matched 색인어
	'''
	import numpy as np
	matched_index_words=[]
	# matched_name_words=[]
	corresponding_keywords_index=[]
	# corresponding_keywords_name=[]
	### get 색인어
	index_words_1 = get_index_words('dataset/index_words.txt')
	index_words_2 = get_index_words('dataset/index_words_.txt')
	bleu_similar_scores_1 = np.zeros(shape=(len(index_words_1),len(keywords)))
	# bleu_similar_scores_2 = np.zeros(shape=(len(index_words_2),len(keywords)))
	cosine_similar_scores = np.zeros(shape=(len(index_words_1),len(keywords)))


	for j, keyword in enumerate(keywords):
		for i, index_word in enumerate(index_words_1):
			### calculate similarity using bleu score
			bleu_similar_scores_1[i,j] = sentence_bleu([list(index_word.replace(' ',''))], list(keyword.replace(' ','')), auto_reweigh=True)

	for idx in np.argwhere(bleu_similar_scores_1 == bleu_similar_scores_1.max()):
		matched_index_words.append(index_words_1[idx[0]])
		corresponding_keywords_index.append(keywords[idx[1]])
	return [matched_index_words, corresponding_keywords_index]#, [matched_name_words, corresponding_keywords_name],


def need_recheck(opinion):
	time_keywords = ['1일']+['1개월', '1달', '일개월', '한달','1 개월', '1 개 월', '1개 월']+['12개월', '12달', '십이개월', '12 개월', '12 개 월', '12개 월']+['3개월', '3달', '삼개월', '세달','3 개월', '3 개 월', '3개 월']+['6개월', '6달', '육개월', '여섯달','6 개월', '6 개 월', '6개 월']+['1년', '2년']
	needed=False
	time_keywords_idx=-1
	for time in time_keywords:
		if time in opinion:
			needed = True
			time_keywords_idx = opinion.index(time)
	return needed, time_keywords_idx

def detect_keywords(doctor_opinions):
	import pandas as pd
	index_name_code = pd.read_excel('index_name_code_mapping.xlsx')

	splitted_opinions = doctor_opinion_split(doctoral_opinion=doctor_opinions)
	index_words = []
	name_words=[]
	code_words=[]
	opinions_sub_opinions=[]
	sub_opinions_keywords=[]
	for idx, opinion in enumerate(splitted_opinions):
		# print(f'opinion: -------- {opinion}')
		sub_opinions = opinion.split('-')
		# sub_opinion=sub_opinions[0]
		for idx_,sub_opinion in enumerate(sub_opinions):
			# print(f'sub_opinion: -------- {sub_opinion}')
			# print(f'opinion --- {opinion}')
			needed, time_keyword_idx = need_recheck(sub_opinion)
			# print(f'time_keyword_idx --- {time_keyword_idx}')

			if needed:
				matched_idx = 0
				matched_keyword=''
				keywords = dease_index_words_extract(sub_opinion.strip())
				# print(f'keywords: -------- {keywords}')
				[matched_index_words, corresponding_keywords_index]= extract_index_words(keywords=keywords)
				# print(f'corresponding_keywords --- {corresponding_keywords}')

				for matched_word, keyword in zip(matched_index_words, corresponding_keywords_index):
					keyword_idx = sub_opinion.index(keyword)
					if keyword_idx < time_keyword_idx:
						# print(f'sub_opinion --- {sub_opinion}')

						# print(f'keyword_idx --- {keyword_idx}')
						# print(matched_keyword)
						if keyword_idx >= matched_idx:
							matched_idx = keyword_idx
							matched_keyword=keyword
				if matched_keyword != '':
					index_word = matched_index_words[corresponding_keywords_index.index(matched_keyword)]
					name_word = str(index_name_code.loc[index_name_code['색인어']==index_word]['질환명'].values[0])
					code_word = str(index_name_code.loc[index_name_code['색인어']==index_word]['질환코드'].values[0])
					index_words.append(index_word)
					name_words.append(name_word)
					code_words.append(code_word)
				#
				# matched_idx = 0
				# matched_keyword = ''
				# for matched_word, keyword in zip(matched_name_words, corresponding_keywords_name):
				# 	keyword_idx = sub_opinion.index(keyword)
				# 	if keyword_idx < time_keyword_idx:
				# 		# print(f'sub_opinion --- {sub_opinion}')
				#
				# 		# print(f'keyword_idx --- {keyword_idx}')
				# 		# print(matched_keyword)
				# 		if keyword_idx >= matched_idx:
				# 			matched_idx = keyword_idx
				# 			matched_keyword=keyword
				# name_words.append(matched_name_words[corresponding_keywords_name.index(matched_keyword)])
				opinions_sub_opinions.append(sub_opinion)
				sub_opinions_keywords.append(keywords)
	return index_words, name_words, code_words, opinions_sub_opinions, sub_opinions_keywords
