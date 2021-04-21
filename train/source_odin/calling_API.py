
from Applications.TextClassification.utils import *
def call_API(cnts):
	'''
	API.py calling using post method.
	'''
	##### using python to call the API with post
	import requests
	api_url = 'http://10.150.21.232:3022/classify'
	svckey = 'Af5HQBAvkv'
	hsptcd = 'hospital_01'  ### hostpital code
	pid = '0001'  ### patient id


	json = {'cnts':str(cnts),'svckey':svckey,'hsptcd':hsptcd,'pid':pid}
	response = None
	try:
		response = requests.post(api_url, json=json, timeout=40)
	except ConnectionError as cnn_e:
		print(ConnectionError('Could not connect to API server, check the configuration'))
	# print(response.json())
	result = response.json()['data']['oicn']
	return result

def load_test_data(path ='processed/in_of_distribution_classes.csv'):
	import pandas as pd
	df = pd.read_csv(path, encoding='cp949')
	df = df[['소견','DSES_CD','DSES_NM']].sample(frac=1)
	return df


def get_data_from_hospital(hospital='hospital_01'):
	try:
		conn = psycopg2.connect(host='10.150.21.232', port='5432', user='locs', password='locslab!', database='test')
	except ConnectionError as e:
		raise ConnectionError('Error when connecting to database server, check the network connection or login information./.')
	cur = conn.cursor()
	query = f"select {hospital}_train.id, {hospital}_train.opinion, {hospital}_train.dses_cd, {hospital}_train.dses_nm " \
			f"from {hospital}_train " \
				f"inner join {hospital}_label " \
				f"on {hospital}_train.dses_nm={hospital}_label.dses_nm; "
			# f"where {hospital}_train.dses_nm <> '대장용종' " \
			#     f"and {hospital}_train.dses_nm <> '경화반';"
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


# if __name__=='__main__':
path1 = 'processed/in_of_distribution_classes.csv'
# path2 = 'source_odin/processed/out_of_distribution_classes.csv'
df = load_test_data(path1)

import pandas as pd
# df = get_data_from_hospital()
count_unk=0
count=0
i=0
print(len(df))
result=[]
#
# a = '''* 뇌혈관 MRA 소견을 참고하시기 바랍니다.
# Brain MRA(TOF)
#
# 1. 뇌동맥류 의심, 좌측 중뇌동맥 횡행부.
#  : 3.7mm 정도의 동맥류성 돌출 의심.
# 2. 그 외 보이는 뇌혈관 근위부에 동맥류, 폐색등의 뚜렷한 이상 소견 없습니다.
#
# 권고 : 신경외과 상담.'''
# print(a)
# i=0
for row in df.itertuples():
	i+=1
	# print(f'calling api line 41')
	label = call_API(cnts=str(row[1]))
	# print(f'calling api line 42')
	#
	cleaned_label = hangul_preprocessing(doctor_opinions=str(row[3]).replace('\n', ' ').strip())
	real_label = ''.join(cleaned_label).replace(' ', '').strip()
	if str(label).replace(' ', '').strip() == real_label:
		count_unk += 1
		result.append([row[1], real_label])

	if count %50 ==0:
		print(f'{count} -- {str(label).replace(" ", "").strip()} -- {real_label}')
	count += 1
	if count >=5000:
		break
# #
# #
# result_df = pd.DataFrame(result, columns=['pred','real'])
# result_df.to_csv('processed/good_pred.csv',encoding='cp949')

print(f"\n=================================================")
print(f'{path1}')
print(f'\n{100 * count_unk / count}%')
print(f"=================================================\n")
#
# if __name__=='__main__':
path1 = 'processed/in_of_distribution_classes.csv'
path2 = 'processed/out_of_distribution_classes.csv'
df = load_test_data(path2)

#
# ### load
# with open('processed/val_dataloader', 'rb') as t_loader_f:
# 	train_dataloader = pickle.load(t_loader_f)
#


import pandas as pd
# df = get_data_from_hospital()
count_unk=0
count=0
i=0
print(len(df))
result=[]
for row in df.itertuples():
	i+=1
	# print(f'calling api line 41')
	label = call_API(cnts=row[1])
	# print(f'calling api line 42')
	#
	# cleaned_label = hangul_preprocessing(doctor_opinions=str(row[3]).replace('\n', ' ').strip())
	# real_label = ''.join(cleaned_label).replace(' ', '').strip()
	result.append([str(label).replace(' ', '').strip(), 'unknown'])
	if str(label).replace(' ', '').strip() == 'unknown':
		count_unk += 1
	if count %50 ==0:
		print(f'{count} -- {str(label).replace(" ", "").strip()} -- unknown')
	count += 1


#
# result_df = pd.DataFrame(result, columns=['pred','real'])
# result_df.to_csv('processed/with_odin_ood.csv',encoding='cp949')

	# if str(label).replace(' ', '').strip() == 'unknown':
	# 	count_unk += 1
	# if count %50 ==0:
	# 	print(f'{count} -- {str(label).replace(" ", "").strip()} -- unknown')
	# count += 1


print(f"\n=================================================")
print(f'{path2}')
print(f'\n{100 * count_unk / count}%')
print(f"=================================================\n")