import sys
import csv
sys.path.insert(0,'AI4AI4Climate')
sys.path.insert(0,'AI4AI4Climate/com')
from com.common import *
import os
import argparse
'''
This is a program to parse data images from Roadway Dataset
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
	print(csv_name)
	check_if_dir_existed_create(dest)
	## Read original desktop or mobile dataset
	#for f in os.listdir(args.source):
	#    print(f)
	extract_exif(args.source, csv_name)
