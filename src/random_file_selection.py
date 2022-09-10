import sys
sys.path.insert(0,'AI4AI4Climate')
sys.path.insert(0,'AI4AI4Climate/com')
from com.common import *
import argparse
'''
This is a program to parse data images from the open CityScapse Dataset
'''

def arg_parser():
	parser = argparse.ArgumentParser(description = 'This is a random selection of images programm')
	parser.add_argument('-source', '--source', required=False, type=str, help='Source of images')
	parser.add_argument('-json', '--json', required=False, type=str, help='Source of json files')
	#parser.add_argument('-dest', '--dest', required=False, type=str, help='Destination to copy')
	parser.add_argument('-n', '--n', required=False, type=str, help='Number of files to copy')
	return parser.parse_args()


if __name__ == '__main__':

	args = arg_parser()
	check_if_dir_existed(args.source)
	check_if_dir_existed(args.json)
	time = getCurrentTime()
	dest = os.path.join(os.path.abspath(os.path.join(args.source, '..')), time)
	csv_name = os.path.join(os.path.abspath(os.path.join(args.source, '..')), time+".csv")
	check_if_dir_existed_create(dest)
	### Read CityScape dataset from the specific folder in image and annotation
	random_selection(args.source, args.json, dest, int(args.n))
