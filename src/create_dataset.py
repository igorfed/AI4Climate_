import argparse
from email.mime import image
import os
import sys
import csv
import pandas as pd
from com.common_packages import check_if_file_existed
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from com.common_packages import check_if_dir_existed
from pathlib import Path
def readTextFile(filename):
	List0 = []
	List1 = []
	f = open(filename, 'r')
	for x in f:
		List0.append(os.path.join(x.strip(),"image" ))
		List1.append(os.path.join(x.strip(),"annotation.csv" ))
	f.close()
	return List0, List1

def arg_parser():
	parser = argparse.ArgumentParser(description = 'copy all dataset into one. Merge CSV')
	parser.add_argument('-source', '--source', required=True, type=str, help='Path to the list')
	parser.add_argument('-dest', '--dest', required=False, type=str, help='Destination of images')
	parser.add_argument('-type', '--type', required=True, type=str, help='Type of the dataset [mobile, desktop, roadway, eu2013 ...]')

def merge_csv():
	pass

if __name__ == '__main__':
	filename = 'list.txt'
	dest = "/media/igofed/SSD_1T/AI4CI/FULLDATASET/FULLDATASET"
	dest_image = os.path.join(dest, 'image')
	check_if_dir_existed(dest, True)
	check_if_dir_existed(dest_image, True)
	
	lst0, lst1 = readTextFile(os.path.join(sys.path[0], filename))
	for list in lst0: print( check_if_dir_existed(list))
	df_list = []
	list_stacked = pd.DataFrame()
	for list in lst1:
		check_if_file_existed(list)
		list_stacked = pd.concat([list_stacked, pd.read_csv(list)])
	
	print(type(list_stacked))
	list_stacked.to_csv(os.path.join(dest, "annotation.csv"))

	print('done')