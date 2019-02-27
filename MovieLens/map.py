import numpy as np
import re
import pickle

node2feature = []
child2father = []

for i in range(20):

	filename = './tree/tree'+str(i)+'.dot'

	with open(filename, "r") as f:
	    line = f.readline()

	    #非叶子结点，通过数字查询特征名
	    node2feature_dict = {}
	    #父子关系
	    child2father_dict = {}
	    while line != None and line != "":
	        line = f.readline()
	        #正则匹配叶子结点或非叶子结点的含义
	        matchObj = re.match( r'(\d+) (\[label=")(\w+) (.+)', line)
	        if matchObj:
	            if matchObj.group(3) != "friedman_mse":
	                node2feature_dict[matchObj.group(1)] = matchObj.group(3)
	        else:
	            #正则匹配:父结点 —> 子结点
	            matchObj = re.match( r'(\d+) -> (\d+) ', line)
	            if matchObj:
	                child2father_dict[matchObj.group(2)] = matchObj.group(1)

	node2feature.append(node2feature_dict)
	print(node2feature_dict)
	child2father.append(child2father_dict)


#保存字典
with open('./tree/node2feature.pkl', 'wb') as f:
	pickle.dump(node2feature, f)

with open('./tree/child2father.pkl', 'wb') as f:
	pickle.dump(child2father, f)



	    