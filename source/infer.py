# -*-coding:utf-8-*-
import os
import shutil

from sklearn.externals import joblib
import numpy as np
from sklearn import decomposition
from _facial_landmarks import find_facial_landmark
from _create_feature import generateAllFeatures
import argparse

img_path = './test_image/29.jpg'


pre_model = joblib.load('../model/face_model.pkl')
features = np.loadtxt('../data/features_ALL.txt', delimiter=',')
pca = decomposition.PCA(n_components=29)
pca.fit(features)

def infer(img_path, plot=True):
	landmarks = find_facial_landmark(img_path,plot=False)
	if len(landmarks) == 0:
		return 0
	print('landmars:',landmarks.shape)
	try:
		my_features = generateAllFeatures(landmarks,training=True)
	except:
		return 0
	predictions = []
	if len(my_features.shape) > 1:
		for i in range(len(my_features)):
			feature = my_features[i, :]
			feature_transfer = pca.transform(feature.reshape(1, -1))
			predictions.append(pre_model.predict(feature_transfer))
		print('照片中的人颜值得分依次为(满分为5分)：')
		k = 1
		for pre in predictions:
			print('第%d个人：' % k, end='')
			print(str(pre)+'分')
			k += 1
	else:
		feature = my_features
		print(feature.shape)
		feature = pca.transform(feature)
		predictions.append(pre_model.predict(feature))
		print('照片中的人颜值得分为(满分为5分)：')
		k = 1
		for pre in predictions:
			print(str(pre)+'分')
			k += 1
	return predictions[0]
if __name__ == '__main__':
	root = 'E:\\face_score\\mingxing_beauty'
	for i,img in enumerate(os.listdir(root)):
		score = infer(os.path.join(root,img),plot=False)[0]
	
