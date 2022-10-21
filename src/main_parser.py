import sys
import csv
import os
from time import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#from com.common import *
from com.common_packages import arg_parser
from com.common_packages import check_if_dir_existed
from com.common_packages import getCurrentTime
from com.common_packages import DATASETS
from com.com_csv import CSV
from com.colors import COLOR
import numpy as np
'''
This is a program to parse data images:
Desktop -> image folder_0 + csv annotation
Mobile  -> image folder_1 + csv annotation
Roadway -> image folder_2 + csv annotation
'''

def destination_path(args):
	'''
	Check if output directory exist 
	'''
	from pathlib import Path
	if args.dest != None:
		'return specified directory'
		return args.dest
	else:
		'return default directory'
		return os.path.join((Path(__file__).parent.parent), 'temp_' + time)

'''
This is a program to parse data images from Roadway Dataset
and creates a new directory with updated csvfile
'''

def duplicates(numbers):
	duplicates = [number for number in numbers if numbers.count(number) > 1]
	return list(set(duplicates))

if __name__ == '__main__':

	args = arg_parser()
	check_if_dir_existed(args.source)
	time = getCurrentTime()
	
	dest = destination_path(args)
	dest = os.path.join(dest, args.type)
	check_if_dir_existed(dest,True)
	dest_images = os.path.join(dest,'image')
	check_if_dir_existed(dest_images, True)
	csv_name = os.path.join(dest, "annotation.csv")
	check_if_dir_existed(dest, True)
	__dataset = DATASETS(source = args.source, dest = dest_images, type = args.type)
	__csv = CSV
	#print(len(__dataset))
	#for i in range(len(__dataset)):
	W, H = [], []
	for i in range(len(__dataset)):

		sample = __dataset[i]
		img = sample['image']
		(height, width, channel)=img.shape
		W.append(width)
		H.append(height)
		#print(f'h: {height}, wight: {width}')
		__dataset.imageCopy2Dest(sample, i)
		##------------------------##
		__csv.name.append(sample['landmarks'][0][0])
		__csv.id.append(f'{i:04n}')
		__csv.hasWater.append(int(sample['landmarks'][0][1]))
		__csv.TimeEvent.append(sample['landmarks'][0][2])
		__csv.lat.append(sample['landmarks'][0][3])
		__csv.lon.append(sample['landmarks'][0][4])
		__csv.dict["dataset"] = __csv.name
		__csv.dict["image_id"] = __csv.id
		__csv.dict["timeEvent"] = __csv.TimeEvent
		__csv.dict["hasWater"] = __csv.hasWater
		__csv.dict["lat"] = __csv.lat
		__csv.dict["lon"] = __csv.lon
	print(COLOR.BBlue)
	print('W', np.max(duplicates(W)))
	print('H', np.max(duplicates(H)))
	#my_dict = {i:H.count(i) for i in H}

	#print('dict H', H)
	print('Has No Water Flooding: {}'.format(__csv.hasWater.count(0)))
	print('Has    Water Flooding: {}'.format(__csv.hasWater.count(1)))
	print('In Total             : {}'.format(len(__csv.hasWater)), COLOR.END)
	
	with open(csv_name, 'w', newline="") as csv_file_patient:
		w = csv.writer(csv_file_patient, delimiter=',')
		w.writerow(__csv.dict.keys())
		w.writerows(zip(*__csv.dict.values()))
	csv_file_patient.close()
	print("CSV Done", csv_name)