import pickle
import numpy as np
from preparing import gbdt_attention, movies_info_split, explain

import warnings, os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#gbdt的字典，需要查找叶子节点的祖先节点
node2feature_file = open('./tree/node2feature.pkl', 'rb')
node2feature = pickle.load(node2feature_file)
child2father_file = open('./tree/child2father.pkl', 'rb')
child2father = pickle.load(child2father_file)

user_id = int(input('Please input you ID:'))

data, attention = gbdt_attention(user_id)

#10部电影的名字和信息
titles, info = movies_info_split(data)


for index in range(10):
	fea_importance = {'sex' : 0, 'age' : 0, 'job' : 0, 'genres0' : 0, 'genres1' : 0\
		, 'genres2' : 0, 'genres3' : 0, 'genres4' : 0, 'genres5' : 0}
	for i in range(20):
		nodes = data[index][:20]
		#当前第i棵树被选中的叶子节点的编号
		leafNode = str(nodes[i]) 
		#当前被选中的叶子节点的祖先节点编号
		parentNodes = []
		#第i棵树的字典
		node2feature_dict = node2feature[i]
		child2father_dict = child2father[i]
		
		#获取祖先节点编号
		node = child2father_dict[leafNode]
		if int(node) == 0:
			parentNodes.append(node)
			break
		else:
			parentNodes.append(node)
			while int(node):
				node = child2father_dict[node]
				parentNodes.append(node)
				if int(node) == 0:
					break
	        
		#查询祖先节点对应的feature，更新fea_importance中的权重
		for node in parentNodes:
			fea_name = node2feature_dict[node]
			fea_importance[fea_name] = fea_importance[fea_name] + attention[i]
            
    #排序
	imp_sort = sorted(fea_importance.items(), key=lambda d: d[1], reverse=True) 
	sort_fea = []
	for fea in imp_sort:
		sort_fea.append(fea[0])
	#删除无意义的genres
	for i in info[index]:
		if info[index][i] == -1:
			sort_fea.remove(i)

	#解释
	explanation = explain([sort_fea[0],sort_fea[1]], [info[index][sort_fea[0]],info[index][sort_fea[1]]])
	print('Recommend the movie', titles[index], 'to you.', explanation)