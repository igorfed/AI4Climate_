import sys
import csv
import glob
import os
from typing import Tuple
from time import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#from com.common import *
import argparse
from com.common_packages import check_if_dir_existed, check_if_file_existed
from com.common_packages import getCurrentTime
from com.common_packages import DATASETS
from com.com_csv import CSV
from com.colors import COLOR
import com.common as common
import numpy as np

'''
This is a program to parse data images:
Desktop -> image folder_0 + csv annotation
Mobile  -> image folder_1 + csv annotation
Roadway -> image folder_2 + csv annotation
'''

def duplicates(numbers):
	duplicates = [number for number in numbers if numbers.count(number) > 1]
	return list(set(duplicates))

def arg_parser():
	parser = argparse.ArgumentParser(description = 'This is a programm to re-annotate sellected dataset [desktop, mobile]')
	parser.add_argument('-s', '--source', 
				required=True, type=str, help='The source of the images')
	parser.add_argument('-d', '--dest', 
				required=False, type=str, help='The destination of the images [default if it is not specified]')
	parser.add_argument('-t', '--type', 
				required=True, type=str, help='Type of the dataset [mobile, desktop, roadway, eu2013 ...]')
	return vars(parser.parse_args())




def main():

	time = getCurrentTime()

	def source_data(source: str)-> Tuple[str, str]:
		'''
		argument: source is a path to the dataset
		dataset 
				|->images
				*.csv
		outputs:
			path2images
			path2csv
		'''
		check_if_dir_existed(source)
		path2image = os.path.join(source, 'images')
		path2csv = []
		items = os.listdir(source)
		for item in items:
			if os.path.isdir(os.path.join(source,item)):
				if os.path.join(source,item)=='images':
					path2images = os.path.join(source,item)

			elif os.path.isfile(os.path.join(source,item)):
				ext = os.path.splitext(item)[-1].lower()
				if ext=='.csv':
					path2csv = os.path.join(source,item)
			else:
				path2image, path2csv = [],[]					
		return 	path2image, path2csv

	def dest_data(dest: str)-> Tuple[str, str]:
		'''
		argument: dest is a path to the dataset
		output 
		dest 
				|->images folder
				*.csv
		'''
		check_if_dir_existed(dest, True)
		path2image = os.path.join(dest, 'images')
		check_if_dir_existed(path2image, True)
		path2csv = os.path.join(dest, 'annotation.csv')
		
		return 	path2image, path2csv
	
	### Check source images and annotations
	args = arg_parser()	
	source_path2image, source_path2csv = source_data(source=args['source'])
	print(f"Path to images : {check_if_dir_existed(source_path2image)}" )
	print(f"Path to csv    : {source_path2csv}" )
	### Create path to destination
	check_if_dir_existed(args['dest'], True)
	dest = common.destination_path(args['dest'], time)
	dest_path2image, dest_path2csv = dest_data(dest=dest)
	print(f"Dest Path to images : {dest_path2image}" )
	print(f"Dest Path to csv    : {dest_path2csv}" )
	check_if_file_existed(source_path2csv)

	__dataset = common.SOURCE_DATASETS(
			source = source_path2image, 
			dest = dest_path2image, 
			type = args['type'], 
			csv_file=source_path2csv)
	W = []
	H = []
	# Parse all images and
	#for i in range(len(__dataset)):
	for i in range(10):
		sample = __dataset[i]

		#print(sample['has_water'], sample['dateTimeEvent'], sample['lat'], sample['lon'])
		#fname = f"desktop_{sample['has_water']}_{sample['dateTimeEvent']}_{sample['lat']}_{sample['lon']}.png"

		#print(fname)
		#__dataset[i]
#		img = sample['image']
#		(height, width, channel)=img.shape
#		W.append(width)
#		H.append(height)
#		print(f'h: {height}, wight: {width}')

	#check_if_dir_existed(args['dest'])
	
	
	#dest = os.path.join(dest, args['type'])
	#dest_images = os.path.join(dest,'image')
	#check_if_dir_existed(dest_images, True)
	#csv_name = os.path.join(dest, f"{args[type]}_annotation.csv")
	#check_if_dir_existed(dest, True)
	#__dataset = DATASETS(source = args.source, dest = dest_images, type = args.type)


if __name__ == '__main__':
	'''
	This is a program to parse data images from Roadway Dataset
	and creates a new directory with updated csvfile
	'''

	
	main()
	print('done')
	
	
	
	
#	__dataset = DATASETS(source = args.source, dest = dest_images, type = args.type)
	#__csv = CSV
	#W, H = [], []
	#for i in range(len(__dataset)):

		#sample = __dataset[i]
		#img = sample['image']
		#(height, width, channel)=img.shape
		#W.append(width)
		#H.append(height)
		#print(f'h: {height}, wight: {width}')
		#__dataset.imageCopy2Dest(sample, i)
		##------------------------##
		#name = sample['landmarks'][0][0]
		#id = f'{i:04n}'
		#hasWater = sample['landmarks'][0][1]

		
		#fname = f'{name}_{i:04n}_{hasWater}.png'

		#print(fname)
		
		#__csv.fname.append(fname)
		#__csv.name.append(sample['landmarks'][0][0])
		#__csv.id.append(f'{i:04n}')
		#__csv.hasWater.append(int(sample['landmarks'][0][1]))
		#__csv.TimeEvent.append(sample['landmarks'][0][2])
		#__csv.lat.append(sample['landmarks'][0][3])
		#__csv.lon.append(sample['landmarks'][0][4])
		#__csv.dict["fname"] = __csv.fname
		#__csv.dict["dataset"] = __csv.name
		#__csv.dict["image_id"] = __csv.id
		#__csv.dict["timeEvent"] = __csv.TimeEvent
		#__csv.dict["hasWater"] = __csv.hasWater
		#__csv.dict["lat"] = __csv.lat
		#__csv.dict["lon"] = __csv.lon
	#print(COLOR.BBlue)
	#print('W', np.max(duplicates(W)))
	#print('H', np.max(duplicates(H)))
	#my_dict = {i:H.count(i) for i in H}

	#print('dict H', H)
	#print('Has No Water Flooding: {}'.format(__csv.hasWater.count(0)))
	#print('Has    Water Flooding: {}'.format(__csv.hasWater.count(1)))
	#print('In Total             : {}'.format(len(__csv.hasWater)), COLOR.END)
	
	#with open(csv_name, 'w', newline="") as csv_file_patient:
	#	w = csv.writer(csv_file_patient, delimiter=',')
		#w.writerow(__csv.dict.keys())
		#w.writerows(zip(*__csv.dict.values()))
	#csv_file_patient.close()
	#print("CSV Done", csv_name)