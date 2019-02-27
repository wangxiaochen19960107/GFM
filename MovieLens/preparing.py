import numpy as np
import pandas as pd
import pickle
from sklearn.externals import joblib
from keras.models import load_model, Model

from mylayers import MyMeanPool, MySumLayer, MyFlatten



#dataframe处理为网络需要的数据格式
def csv_processing(df):
	users_id = df['user_id'].values
	items_id = df['item_id'].values
	sexes = df['sex'].values
	ages = df['age'].values
	jobs = df['job'].values

	genres = np.zeros((6,len(df['genres0'])), dtype='i8')
	genres[0] = np.array(df['genres0'])
	genres[1] = np.array(df['genres1'])
	genres[2] = np.array(df['genres2'])
	genres[3] = np.array(df['genres3'])
	genres[4] = np.array(df['genres4'])
	genres[5] = np.array(df['genres5'])
	genres = genres.T

	if 'rate' in df.columns:
		probs = df['rate'].values
		probs = np.array([((x>0)-0) for x in probs])
		return users_id, items_id, sexes, ages, jobs, genres, probs
	else:
		return users_id, items_id, sexes, ages, jobs, genres



#用户曾经看过的电影
def user_consumed(id):
	users_seen = pd.read_csv('./data/ratings.csv', engine='python')
	seen = users_seen[users_seen['user_id']==id]['index'].values
	return seen


#预测的用户没看过的top10,返回10部电影的全部数据、attention
def gbdt_attention(id):
	#数据下标从0开始
	id = id - 1
	users_info = pd.read_csv('./data/users.csv', engine='python')
	movies_info = pd.read_csv('./data/movies.csv', engine='python')
	#当前用户信息
	test_user = users_info.loc[id]

	#所有电影信息后添加指定用户的信息
	user_cols = users_info.columns
	for col in user_cols:
	    movies_info[col] = test_user[col]

	#准备gbdt需要的数据，然后载入gbdt模型对数据编码
	gbdt_cols = ['sex','age','job','genres0','genres1','genres2','genres3','genres4','genres5']
	gbdt_data = movies_info[gbdt_cols]
	gbdt_20 = joblib.load('./model/gbdt.pkl')
	gbdt_output = gbdt_20.apply(gbdt_data)

	#deepfm+gbdt需要的数据
	tree_cols = []
	for i in range(20):
	    arr = []
	    for j in range(len(gbdt_output)):
	        arr.append(int(gbdt_output[j][i][0]))
	    col_name = 'tree'+str(i)
	    tree_cols.append(col_name)
	    arr = (np.array(arr)).reshape(len(gbdt_output),1)
	    movies_info[col_name] = arr

	genres_cols = ['genres0','genres1','genres2','genres3','genres4','genres5']
	single_cols = tree_cols+['user_id','item_id','sex','age','job']

	data = []
	for col in single_cols:
		data.append(np.array(movies_info[col].values).reshape(len(gbdt_output),1))

	genres = np.zeros((6,len(movies_info['genres0'])), dtype='i8')
	genres[0] = np.array(movies_info['genres0'])
	genres[1] = np.array(movies_info['genres1'])
	genres[2] = np.array(movies_info['genres2'])
	genres[3] = np.array(movies_info['genres3'])
	genres[4] = np.array(movies_info['genres4'])
	genres[5] = np.array(movies_info['genres5'])
	genres = genres.T

	data.append(genres)

	#载入训练好的模型，预测前10，得到下标，返回10部电影的数据
	model = load_model('./model/model.h5', custom_objects={"MyMeanPool" : MyMeanPool, \
		"MySumLayer" : MySumLayer, "MyFlatten" : MyFlatten})
	pred = model.predict(data)
	seen = user_consumed(id)
	not_seen = np.delete(pred,seen)
	top10_index = np.argsort(not_seen)[:10]

	titles = np.array(movies_info['title']).reshape(len(gbdt_output),1)

	selected_movie_data = []
	for index in top10_index:
	    arr = []
	    for col in data[:25]:
	        arr.append(col[index][0])
	    arr.append(data[25][index])
	    arr.append(titles[index][0])
	    selected_movie_data.append(arr)

    #取attention层的输出作为输出新建为attention_model,预测后的输出为20棵叶子节点的重要性
	attention_model = Model(inputs=model.input,outputs=model.get_layer('attention').output)
	attention_output = attention_model.predict(data)

	return selected_movie_data, attention_output[0][0]


#对recommendation list的信息进行分割
def movies_info_split(data):
	titles = []
	info = [{'sex':0,'age':0,'job':0,'genres0':0,'genres1':0,'genres2':0,'genres3':0,'genres4':0,'genres5':0},
	       {'sex':0,'age':0,'job':0,'genres0':0,'genres1':0,'genres2':0,'genres3':0,'genres4':0,'genres5':0},
	       {'sex':0,'age':0,'job':0,'genres0':0,'genres1':0,'genres2':0,'genres3':0,'genres4':0,'genres5':0},
	       {'sex':0,'age':0,'job':0,'genres0':0,'genres1':0,'genres2':0,'genres3':0,'genres4':0,'genres5':0},
	       {'sex':0,'age':0,'job':0,'genres0':0,'genres1':0,'genres2':0,'genres3':0,'genres4':0,'genres5':0},
	       {'sex':0,'age':0,'job':0,'genres0':0,'genres1':0,'genres2':0,'genres3':0,'genres4':0,'genres5':0},
	       {'sex':0,'age':0,'job':0,'genres0':0,'genres1':0,'genres2':0,'genres3':0,'genres4':0,'genres5':0},
	       {'sex':0,'age':0,'job':0,'genres0':0,'genres1':0,'genres2':0,'genres3':0,'genres4':0,'genres5':0},
	       {'sex':0,'age':0,'job':0,'genres0':0,'genres1':0,'genres2':0,'genres3':0,'genres4':0,'genres5':0},
	       {'sex':0,'age':0,'job':0,'genres0':0,'genres1':0,'genres2':0,'genres3':0,'genres4':0,'genres5':0}]

	for i,d in enumerate(data):
		titles.append(d[-1])
		info[i]['sex'] = d[22]
		info[i]['age'] = d[23]
		info[i]['job'] = d[24]
		for j,dd in enumerate(d[25]):
			if dd==0:
				info[i]['genres'+str(j)] = -1
			else:
				info[i]['genres'+str(j)] = dd

	return titles, info



def explain(reasons, data):
	int2genres = ['' ,'Horror', 'Drama', 'Musical', 'Western', 'Sci-Fi', 'Romance', 'Action', \
	'Documentary', 'Comedy', 'Animation', "Film-Noir", 'Mystery',"Children's", 'Fantasy', \
	'Crime', 'Adventure', 'War', 'Thriller']
	int2age = [1, 18, 25, 35, 45, 50 ,56]
	int2sex = ['Female', 'Male']
	int2job = ["not Specified", "academic/educator", "artist", "clerical/admin", "college/grad student", \
		"customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker", \
		"K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist", \
		"self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]

	explanation =''

	for i , reason in enumerate(reasons):
		if reason == 'age':
			explanation = explanation + 'You age is ' + str(int2age[data[i]]) + '. '
		if reason == 'sex':
			explanation = explanation + 'You sex is ' + str(int2sex[data[i]]) + '. '
		if reason == 'job':
			explanation = explanation + 'You job is ' + str(int2job[data[i]]) + '. '
		if 'genres' in reason:
			explanation = explanation + 'The movie is about ' + str(int2genres[data[i]]) + '. '

	return explanation


