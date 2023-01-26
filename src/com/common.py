import os
from com.colors import COLOR
import numpy as np
import pandas as pd

def destination_path(
					out : str, 
					time : str) -> str:
	'''
	Check if output directory exist 
	'''
	from pathlib import Path
	if out != None:
		'return specified directory'
		return out
	else:
		'return default directory'
		return os.path.join((Path(__file__).parent.parent.parent), 'temp_' + time)


class SOURCE_DATASETS():
	
	def __init__(self, source, dest, type, csv_file):
		t = COLOR.Green + 80 * '=' + COLOR.END
		self.valid_img = [".jpg", "jpeg", ".png"]
		print(t)
		"""
		root_dir is the path to dataset
		output is the output dir for renamed images
		"""
		self.type = type
		self.source_dir = source
		self.dest_dir = dest
		self.files2list()
		self.landmarks_frame = pd.read_csv(csv_file) 
		
		self.dict_CSV = {}
		self.copy_fname = self.fname.copy()

	def files2list(self):
		"""
		create a list of all image names
		"""
		
		self.fname = []
		for _, fname in enumerate(os.listdir(self.source_dir)):
			fname = os.path.join(self.source_dir, fname)
			if os.path.splitext(fname)[-1].lower() not in self.valid_img:
				continue
			self.fname.append(fname)
		

	
	def read_csv(self, i : int)-> dict():
		
		def isnan(x):
			return 0 if np.isnan(x) else x

		dateTimeEvent = self.landmarks_frame.iloc[i,0]
		################### lat, lon ###########################
		lat = int(float(self.landmarks_frame.iloc[i,1])*1e4)/1e4
		lon = int(float(self.landmarks_frame.iloc[i,2])*1e4)/1e4
		################### id #################################
		id = int(self.landmarks_frame.iloc[i,3])
		water_detail1 = int(isnan(self.landmarks_frame.iloc[i,4]))
		water_detail2 = int(isnan(self.landmarks_frame.iloc[i,5]))
		ground_detail = int(isnan(self.landmarks_frame.iloc[i,6]))
		has_water = int(isnan(self.landmarks_frame.iloc[i,7]))
		water_detail3 = int(isnan(self.landmarks_frame.iloc[i,9]))

		sample = {
			'dateTimeEvent': dateTimeEvent , 
			'lat': lat, 
			'lon': lon, 
			'id': id,
			'water_detail1' : water_detail1,
			'water_detail2' : water_detail2,
			'water_detail3' : water_detail3,
			'ground_details' : ground_detail,
			'has_water' : has_water}
		print(i, np.size(sample))
		return sample

	def read_image(self, i : int, csv_sample: dict()):

		fname = f"desktop_{csv_sample['has_water']}_{csv_sample['dateTimeEvent']}"
		
		for i in range(len(self.copy_fname)):
			if fname in self.copy_fname[i]:
				if str(csv_sample['lat']) in self.copy_fname[i]:
					if str(csv_sample['lon']) in self.copy_fname[i]:
						print(len(self.copy_fname), idx, self.copy_fname[i])
						image_name = self.copy_fname[i]

						return image_name

		
	def __len__(self):
		return len(self.landmarks_frame)


	def __getitem__(self, idx):

		def image_name(fname):
			name = fname.split('_')[0]
			hasWater = fname.split('_')[1]
			TimeEvent = fname.split('_')[2]
			lat = format(float(fname.split('_')[3]), ".4f")
			lon = format(float(fname.split('_')[4]), ".4f")
			return name, hasWater, TimeEvent, lat, lon

		def image_name_roadway(fname):
			"image_0.png"
			stat = os.stat(fname)
			TimeEvent = int(stat.st_mtime)*1000 +idx
			lat = '12.3456'
			lon = '78.9012'
			return "roadway", 1,  TimeEvent, lat, lon
		def image_name_cityscape(fname, tp):
			"image_0.png"
			stat = os.stat(fname)
			TimeEvent = int(stat.st_mtime)*1000 +idx
			lat = '12.3456'
			lon = '78.9012'
			return tp, 0,  TimeEvent, lat, lon


		#fname = self.fname[idx]
		#name, _ =  os.path.splitext(fname)
		
		csv_sample = self.read_csv(idx)
		sample = self.read_image(idx, csv_sample)

		print('sample',csv_sample)
		#print(idx, 'id', csv_sample['id'], csv_sample['dateTimeEvent'], csv_sample['water_detail1'] )
		#print(fname)
		#if f"desktop_{csv_sample['has_water']}" in self.copy_fname[0]:
		#	lat = format(float(csv_sample['lat']), ".4f")
		#	if str(lat) in self.copy_fname[0]:
		#for i in range(len(self.copy_fname)):
			#if fname in self.copy_fname:
				
		#		print(idx, fname, self.copy_fname[0])
		#print(fname)
		#try:
		#	f = open(fname)
		#except IOError:
		#	print(COLOR.Red + f'fname{fname}File not accessible' + COLOR.END)
		#finally:
		#	f.close()
		#image = Image.open(fname).convert('RGB')
		#image.verify()
		#image = np.asarray(image)
		#image[0] = (image[0] - np.min(image[0])) / (np.max(image[0]) - np.min(image[0]))
		#image[1] = (image[1] - np.min(image[1])) / (np.max(image[1]) - np.min(image[1]))
		#image[2] = (image[2] - np.min(image[2])) / (np.max(image[2]) - np.min(image[2]))

		#if self.type in ["desktop", "mobile"]:
	#		landmarks = np.array(image_name(os.path.basename(name)))
	#	elif self.type =="roadway":
	#		landmarks = np.array(image_name_roadway(fname))
	#	elif self.type in ["berlin", "bonn", "munich","mainz", "zurich" ]:
	#		landmarks = np.array(image_name_cityscape(fname, self.type))
	#	else:
	#		print(COLOR.Red + "User set-up a wrong argument in type" + COLOR.END)	
	#		exit
		
	#	landmarks = landmarks.reshape(-1, 5)
		
	#	sample = {'image': image, 'landmarks': landmarks}
		#print(idx, fname, image.size, image.shape, type(image))
	#	print(COLOR.Green,'ID\t: {} in {}'.format(idx, len(self.fname)))
	#	print('Fname\t: {}'.format(fname))
	#	print('Shape\t: {}'.format(image.shape), COLOR.END)
	#	print(sample['landmarks'][0][0], sample['landmarks'][0][1], sample['landmarks'][0][2], sample['landmarks'][0][3], sample['landmarks'][0][3])
	#
		sample = []	
		return sample
