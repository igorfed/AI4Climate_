import sys
import csv
from time import time
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from com.common_packages import check_if_dir_existed
from com.common_packages import check_if_file_existed
from com.common_packages import getCurrentTime
import os
import argparse
from com.common_packages import random_selection
from com.common_packages import DATASETS
from com.com_csv import CSV
'''
This is a program to parse data images from the open CityScapse Dataset
'''

def arg_parser():
	parser = argparse.ArgumentParser(description = 'This is a random selection of images programm')
	parser.add_argument('-source', '--source', required=False, type=str, help='Source of images')
	parser.add_argument('-dest', '--dest', required=False, type=str, help='Destination to copy')
	parser.add_argument('-type', '--type', required=True, type=str, help='Type of the dataset [mobile, desktop, roadway, eu2013 ...]')
	parser.add_argument('-n', '--n', required=False, type=str, help='Number of files to copy')
	return parser.parse_args()

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

if __name__ == '__main__':

	args = arg_parser()
	check_if_dir_existed(args.source)
	time = getCurrentTime()
	dest = destination_path(args)
	check_if_dir_existed(dest,True)
	dest_images = os.path.join(dest,'image')
	check_if_dir_existed(dest_images, True)
	csv_name = os.path.join(dest, "annotation.csv")
	check_if_dir_existed(dest, True)
	__dataset = DATASETS(source = args.source, dest = dest_images, type = args.type)
	__csv = CSV
	print(len(__dataset))
	lst = range(0,len(__dataset)-1)
	print(lst, len(lst))
	import random
	lst =  random.sample(lst, k = int(args.n))
	print('lst', lst, len(lst))
	idx = 0
	for i in range(len(lst)):
		k = lst[i]
		sample = __dataset[k]
		#print(idx, k, i, sample['landmarks'][0][0], sample['landmarks'][0][1], sample['landmarks'][0][2], sample['landmarks'][0][3], sample['landmarks'][0][3])
		

		name = sample['landmarks'][0][0]
		hasWater = sample['landmarks'][0][1]
		fname = f'{name}_{idx:04n}_{hasWater}.png'
		

		__csv.fname.append(fname)
		__csv.name.append(sample['landmarks'][0][0])
		__csv.id.append(f'{idx:04n}')
		__csv.hasWater.append(sample['landmarks'][0][1])
		__csv.TimeEvent.append(sample['landmarks'][0][2])
		__csv.lat.append(sample['landmarks'][0][3])
		__csv.lon.append(sample['landmarks'][0][4])
		#print(idx, k, i)
		__dataset.imageCopy2Dest(sample, idx)
		idx = idx +1

	__csv.dict["fname"] = __csv.fname		
	__csv.dict["dataset"] = __csv.name
	__csv.dict["image_id"] = __csv.id
	__csv.dict["timeEvent"] = __csv.TimeEvent
	__csv.dict["hasWater"] = __csv.hasWater
	__csv.dict["lat"] = __csv.lat
	__csv.dict["lon"] = __csv.lon
	print('water:', __csv.hasWater.count(0), ' in Total:', len(__csv.hasWater))
	with open(csv_name, 'w', newline="") as csv_file_patient:
		w = csv.writer(csv_file_patient, delimiter=',')
		w.writerow(__csv.dict.keys())
		w.writerows(zip(*__csv.dict.values()))
	csv_file_patient.close()
	print("CSV Done", csv_name)