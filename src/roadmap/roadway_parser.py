import sys
import csv
import os
from time import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from com.common import *
'''
This is a program to parse data images from the open the Desktop and Mobile datasets
and creates a new directory with updated csv file
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
		return os.path.join((Path(__file__).parent.parent.parent), 'temp_' + time)

'''
This is a program to parse data images from Roadway Dataset
and creates a new directory with updated csvfile
'''

if __name__ == '__main__':

	args = arg_parser()
	check_if_dir_existed(args.source)
	time = getCurrentTime()
	dest = destination_path(args)
	check_if_dir_existed(dest,True)
	dest_images = os.path.join(dest,'image')
	check_if_dir_existed(dest_images, True)
	csv_name = os.path.join(dest, time+".csv")
	check_if_dir_existed_create(dest)
	__dataset = DESKTOP(source = args.source, dest = dest_images, type = "roadway")
	__csv = CSV
	print(len(__dataset))
	for i in range(len(__dataset)):
		sample = __dataset[i]
		print(sample['landmarks'][0][0], sample['landmarks'][0][1], sample['landmarks'][0][2], sample['landmarks'][0][3], sample['landmarks'][0][3])
		__dataset.imageCopy2Dest(sample, i)
		##------------------------##
		__csv.name.append(sample['landmarks'][0][0])
		__csv.id.append(f'{i:04n}')
		__csv.hasWater.append(sample['landmarks'][0][1])
		__csv.TimeEvent.append(sample['landmarks'][0][2])
		__csv.lat.append(sample['landmarks'][0][3])
		__csv.lon.append(sample['landmarks'][0][4])
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