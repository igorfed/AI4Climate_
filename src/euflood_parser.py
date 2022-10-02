import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from com.common_packages import arg_parser_eu
	
from com.common_packages import check_if_dir_existed
from com.common_packages import check_if_file_existed
from com.com_csv import CSV
from com.common_packages import getCurrentTime
from com.common_packages import currentTime2Millisec
from com.colors import COLOR
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv
'''
This is a program to parse eu-flood-2013-small-dataset:
Args: source - path to the folder with dataset
	: flooding - path to the text file with flooding
	#: irrelevant - path to the text file with irrelevant images for non flooding
	: metadata - path to the json file with images' metadata
	: out - path to the output folder (will create : 0 - folder with all images without flooded images, 
													 1 - folder with all images with flooded images,
													 *.csv annotation file
													 all - folder with flooded non flooded images  
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



class EU2013():
	def __init__(self, source, out):
		"""
		source is the path to dataset
		output is the output dir for renamed images
		"""
		t = COLOR.Green + 80 * '=' + COLOR.END
		self.valid_img = [".jpg", "jpeg"]
		print(t)
		self.source_dir = source
		self.dest_dir = out
		self.files2list()
		#self.dict_CSV = {}

	def files2list(self):
		self.fname = []
		for _, fname in enumerate(os.listdir(self.source_dir)):
			fname = os.path.join(self.source_dir, fname)
			if os.path.splitext(fname)[-1].lower() not in self.valid_img:
				continue
			self.fname.append(fname)

	def __len__(self):
		return len(self.fname)

	def readTextFile(self, filename):
		List = []
		f = open(filename, 'r')
		for x in f:
			List.append(x.strip())
		f.close()
		return List

	def __getitem__(self, idx):
		fname = self.fname[idx]
		name, _ =  os.path.splitext(fname)
		try:
			f = open(fname)
		except IOError:
			print(f'fname{name}File not accessible')
		finally:
			f.close()

		image = Image.open(fname).convert('RGB')
		image.verify()
		image = np.asarray(image)
		name = os.path.basename(name)
		sample = {'image': image, 'name': name}
		return sample

	
if __name__ == '__main__':
	dataset= 'eu2013'
	args = arg_parser_eu()
	check_if_dir_existed(args.source)
	check_if_file_existed(args.flooding)
	time = getCurrentTime()
	dest = destination_path(args)
	dest = os.path.join(dest, dataset)
	check_if_dir_existed(dest,True)
	dest_images = os.path.join(dest,'image')
	check_if_dir_existed(dest_images, True)
	csv_name = os.path.join(dest, "annotation.csv")
	check_if_dir_existed(dest, True)
	__dataset = EU2013(source = args.source, out = args.dest)
	flooding = __dataset.readTextFile(filename=args.flooding)
	__csv = CSV
	for i in range(len(__dataset)):
		sample = __dataset[i]
		if (sample['name'] in flooding):
			#__dataset.imageCopy2Dest(sample, i)
			__csv.name.append(dataset)
			__csv.id.append(f'{i:04n}')
			__csv.hasWater.append(1)
			__csv.TimeEvent.append(currentTime2Millisec())
			__csv.lat.append('12.3456')
			__csv.lon.append('78.9012')
			r = f'{__csv.name[-1]}_{i:04n}_{__csv.hasWater[-1]}.png'
			print(r)
			image = Image.fromarray(sample['image'])
			image.save(os.path.join(dest_images,r))
	__csv.dict["dataset"] = __csv.name
	__csv.dict["image_id"] = __csv.id
	__csv.dict["timeEvent"] = __csv.TimeEvent
	__csv.dict["hasWater"] = __csv.hasWater
	__csv.dict["lat"] = __csv.lat
	__csv.dict["lon"] = __csv.lon
	print(COLOR.BBlue)
	print('Has No Water Flooding: {}'.format(__csv.hasWater.count(0)))
	print('Has    Water Flooding: {}'.format(__csv.hasWater.count(1)))
	print('In Total             : {}'.format(len(__csv.hasWater)), COLOR.END)
	with open(csv_name, 'w', newline="") as csv_file_patient:
		w = csv.writer(csv_file_patient, delimiter=',')
		w.writerow(__csv.dict.keys())
		w.writerows(zip(*__csv.dict.values()))
		
	csv_file_patient.close()
	print('done')


