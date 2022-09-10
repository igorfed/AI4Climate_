import argparse
#from mobile_parser import FrameLandmarksDataset
import torch
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import csv
import sys
sys.path.insert(0,'AI4AI4Climate')
sys.path.insert(0,'AI4AI4Climate/com')
from com.common import *
from com.datasets import *


def arg_parser():
	parser = argparse.ArgumentParser(description = 'Say hello')
	parser.add_argument('-r', '--r', required=False, type=str, help='root directory')
	parser.add_argument('-c', '--c', required=False, type=str, help='annotation file')
	parser.add_argument('-d', '--d', required=False, type=str, help='the new data based on annotation')

	return parser.parse_args()


class DATASET():

	def __init__(self, root_dir, csv_file, dest):
		t = COLOR.Green + 80*'=' + COLOR.END
		print(t)
		"""
		root_dir is the path to dataset
		csv_file annotation file name
		output is the output dir for renamed images
		data_source is the name of the dataset (same as folder name)
		"""

		self.path2dataset = root_dir
		self.path2csv = csv_file
		self.dest = dest

		self.landmarks_frame = pd.read_csv(csv_file)


	def __len__(self):
		return len(self.landmarks_frame)

	def checkDesktopAnnotation(self):
		tst = []
		for i in range(self.__len__()):
			img_id_ = self.landmarks_frame.iloc[i,0]
			img_TiE = self.landmarks_frame.iloc[i,1]
			img_lat  = format(float(self.landmarks_frame.iloc[i,2]), ".4f")
			img_lon = format(float(self.landmarks_frame.iloc[i,3]), ".4f")
			img_hW_ = int(self.landmarks_frame.iloc[i,4])
			image_name = f'{img_id_}_{img_hW_}_{img_TiE}_{img_lat}_{img_lon}.png'
			tst.append(image_name)
			check_if_file_existed(os.path.join(self.path2dataset,image_name))

			im = Image.open(os.path.join(self.path2dataset,image_name))
			print((self.__len__()), i, 'image_name', image_name)
			im.close()
		z = 0
		for elem in tst:
			if tst.count(elem)>1:

				print(elem, z, tst.index(elem))
				z = z +1

	def reWriteMobileCSV(self, csv_name):
		dict_CSV = {}
		img_id_ = []
		img_TiE = []
		img_lat = []
		img_lon = []
		img_hW_ = []

		for i in range(self.__len__()):
			#img_id_.append(int(self.landmarks_frame.iloc[i,0]))
			img_id_.append(self.landmarks_frame.iloc[i,0])
			img_TiE.append(self.landmarks_frame.iloc[i,1])
			#img_lat.append(self.landmarks_frame.iloc[i,2])
			#img_lon.append(self.landmarks_frame.iloc[i,3])
			#img_hW_.append(self.landmarks_frame.iloc[i,4])
			#print(i, img_id_[-1], img_TiE[-1], img_lat[-1], img_lon[-1])
			#image_name = f'{img_id_[-1]}_{img_hW_[-1]}_{img_TiE[-1]}_{img_lat[-1]}_{img_lat[-1]}.jpg'
			#check_if_file_existed(os.path.join(self.path2dataset,image_name))
			#im = Image.open(os.path.join(self.path2dataset,image_name))
			#print('image_name', image_name)
			img_lat.append(format(float(self.landmarks_frame.iloc[i,2]), ".4f"))
			img_lon.append(format(float(self.landmarks_frame.iloc[i,3]), ".4f"))
			img_hW_.append(int(self.landmarks_frame.iloc[i,4]))

		dict_CSV['image_id'] = img_id_
		dict_CSV['dTimeEvt'] = img_TiE
		dict_CSV['image_lt'] = img_lat
		dict_CSV['image_ln'] = img_lon
		dict_CSV['image_hW'] = img_hW_

		csv_name = os.path.join(self.path2dataset, csv_name)
		with open(csv_name, 'w', newline="") as csv_file:
			w = csv.writer(csv_file, delimiter=',')
			w.writerow(dict_CSV.keys())
			w.writerows(zip(*dict_CSV.values()))
			csv_file.close()

		print("CSV Done", csv_name)

	def reNameMobileImages(self):
		i = 0
		for f in os.listdir(self.path2dataset):
			name , extension = os.path.splitext(f)
			if extension == '.jpg':
				im = Image.open(os.path.join(self.path2dataset,f))
				mobile = name.split('_')[0]
				hw = name.split('_')[1]
				t = name.split('_')[2]
				lat = format(float(name.split('_')[3]), ".4f")
				lon = format(float(name.split('_')[4]),".4f")
				#new_fname = f'{i:04n}_{mobile}_{hw}_{t}_{lat}_{lon}.png'
				new_fname = f'{mobile}_{hw}_{t}_{lat}_{lon}.png'
				im.save(os.path.join(self.path2dataset, new_fname))
				print(new_fname)
				i = i +1
				im.close()





#def Read_all_jpg_in_folder():
#	path= "/media/igofed/SSD_1T/AI4CI/Carlo/data/"
#	for f in os.listdir(path):
#		print(f)
#		name , extension = os.path.splitext(f)
#		im = Image.open(os.path.join(path,f))
#		mobile = name.split('_')[0]
#		hw = name.split('_')[1]
#		t = name.split('_')[2]
#		lat = float(name.split('_')[3])
#		lon = float(name.split('_')[4])
#		lat = format(lat,".4f")
#		lon = format(lon,".4f")
#		new_fname = f'{mobile}_{hw}_{t}_{lat}_{lon}.png'
#		im.save(os.path.join(path, new_fname))
#		im.close()




if __name__ == '__main__':

	args = arg_parser()

	check_if_dir_existed(args.r)
	check_if_file_existed(args.c)
	check_if_dir_existed_create(args.d)
	__dataset = DATASET(root_dir=args.r, csv_file=args.c, dest=args.d)
	## Uncoment to rename images
	#__dataset.reNameMobileImages()
	__dataset.checkDesktopAnnotation()
	#__dataset.reWriteMobileCSV(csv_name='desctop_1111.csv')
	#__dataset.reNameMobileImages()
	#__frames = FrameLandmarksDataset(root_dir = args.r, csv_file = args.c, transform=None)
	#__flood_dataset = FrameLandmarksDataset(root_dir=args.r, csv_file=args.c, img_type='mobile', transform=None)

	#Read_all_jpg_in_folder()
	#print(len(__dataset))
	#__dataset.reWriteMobileCSV(csv_name='desktop2.csv')
	#for i in range(len(__frames)):
#		sample = __frames[i]
#		print(i, sample['image'].size(), sample['landmarks'])
		#print(i)


	print('done')
