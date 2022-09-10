import sys
import csv
sys.path.insert(0,'AI4AI4Climate')
sys.path.insert(0,'AI4AI4Climate/com')
from com.common import *
import argparse
'''
This is a program to parse data images from the open the Desktop and Mobile datasets
and creates a new directory with updated csvfile
'''

def arg_parser():
	parser = argparse.ArgumentParser(description = 'This is a random selection of images programm')
	parser.add_argument('-source', '--source', required=False, type=str, help='Source of images')
	return parser.parse_args()

if __name__ == '__main__':

	args = arg_parser()
	check_if_dir_existed(args.source)
	time = getCurrentTime()
	dest = os.path.join(os.path.abspath(os.path.join(args.source, '..')), time)
	csv_name = os.path.join(os.path.abspath(os.path.join(args.source, '..')), time+".csv")
	check_if_dir_existed_create(dest)
	## Read original desktop or mobile dataset
	__dataset = DESKTOP(source = args.source, dest = dest)
	dict_CSV = {}
	name = []
	id = []
	hasWater = []
	TimeEvent = []
	lat = []
	lon = []
	for i in range(len(__dataset)):
		sample = __dataset[i]
		__dataset.imageCopy2Dest(sample, i)
		##------------------------##
		name.append(sample['landmarks'][0][0])
		id.append(f'{i:04n}')
		hasWater.append(sample['landmarks'][0][1])
		TimeEvent.append(sample['landmarks'][0][2])
		lat.append(sample['landmarks'][0][3])
		lon.append(sample['landmarks'][0][4])
	dict_CSV["dataset"] = name
	dict_CSV["image_id"] = id
	dict_CSV["TimeEvent"] = TimeEvent
	dict_CSV["hasWater"] = hasWater
	dict_CSV["lan"] = lat
	dict_CSV["lon"] = lon
	print('water:', hasWater.count(0), ' in Total:', len(hasWater))
	with open(csv_name, 'w', newline="") as csv_file_patient:
		w = csv.writer(csv_file_patient, delimiter=',')
		w.writerow(dict_CSV.keys())
		w.writerows(zip(*dict_CSV.values()))
	csv_file_patient.close()
	print("CSV Done", csv_name)
