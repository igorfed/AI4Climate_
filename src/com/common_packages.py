from email.mime import image
import os
import argparse
from com.colors import COLOR
from PIL import Image
from skimage import io, transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def arg_parser():
	parser = argparse.ArgumentParser(description = 'This is a random selection of images program')
	parser.add_argument('-source', '--source', required=True, type=str, help='Source of images')
	parser.add_argument('-dest', '--dest', required=False, type=str, help='Destination of images')
	parser.add_argument('-type', '--type', required=True, type=str, help='Type of the dataset [mobile, desktop, roadway, eu2013 ...]')

	return parser.parse_args()

def arg_parser_eu():
    parser = argparse.ArgumentParser(description = 'This is a program to pars and annotate eu-flood-2013-small images')
    parser.add_argument('-source', '--source', required=False, type=str, help='path to the folder with dataset')
    parser.add_argument('-flooding', '--flooding', required=False, type=str, help='path to flooding.txt')
    #parser.add_argument('-irrelevant', '--irrelevant', required=False, type=str, help='path to irrelevant.txt')
    #parser.add_argument('-metadata', '--metadata', required=False, type=str, help='path to metadata.json')
    parser.add_argument('-dest', '--dest', required=False, type=str, help='path to the output folder')
    return parser.parse_args()



def check_if_file_existed(filename):
	if os.path.isfile(filename):
		print(COLOR.Blue + f'filename \t: {filename} existed' + COLOR.END)
	else:
		print(COLOR.Red + f'filename \t: {filename} is not existed' + COLOR.END)

def check_if_dir_existed(dir_name, create=False):
	if not os.path.exists(dir_name):
		print(COLOR.Red +f'folder \t\t: {dir_name} is not available' + COLOR.END)
		if create:
			os.mkdir(dir_name)
			print(COLOR.Green + f'folder \t\t: {dir_name} created' + COLOR.END)	
	else:
		print(COLOR.BBlue + f'folder \t\t: {dir_name} is available' + COLOR.END)

def getCurrentTime():
	from datetime import datetime
	now = datetime.now()
	return now.strftime("%Y_%m_%d_%H_%M_%S")

def currentTime2Millisec():
	'''
	Getting the current time in format YYYY MM DD HH MM SS
	and transfrom it into millisec
	'''
	from datetime import datetime
	curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
	dt_obj = datetime.strptime(str(curr_datetime), '%Y-%m-%d %H-%M-%S')
	millisec = dt_obj.timestamp()*1000
	return int(millisec)

class DATASETS():

	def __init__(self, source, dest, type):
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
		self.dict_CSV = {}

	def files2list(self):
		self.fname = []
		for i, fname in enumerate(os.listdir(self.source_dir)):
			fname = os.path.join(self.source_dir, fname)
			# print(i, os.path.splitext(fname)[-1].lower())
			if os.path.splitext(fname)[-1].lower() not in self.valid_img:
				continue
			self.fname.append(fname)

	def __len__(self):
		return len(self.fname)

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


		fname = self.fname[idx]
		name, _ =  os.path.splitext(fname)
		try:
			f = open(fname)
		except IOError:
			print(COLOR.Red + f'fname{fname}File not accessible' + COLOR.END)
		finally:
			f.close()
		image = Image.open(fname).convert('RGB')
		image.verify()
		image = np.asarray(image)
		#image[0] = (image[0] - np.min(image[0])) / (np.max(image[0]) - np.min(image[0]))
		#image[1] = (image[1] - np.min(image[1])) / (np.max(image[1]) - np.min(image[1]))
		#image[2] = (image[2] - np.min(image[2])) / (np.max(image[2]) - np.min(image[2]))

		if self.type in ["desktop", "mobile"]:
			landmarks = np.array(image_name(os.path.basename(name)))
		elif self.type =="roadway":
			landmarks = np.array(image_name_roadway(fname))
		elif self.type in ["berlin", "bonn", "munich","mainz", "zurich" ]:
			landmarks = np.array(image_name_cityscape(fname, self.type))
		else:
			print(COLOR.Red + "User set-up a wrong argument in type" + COLOR.END)	
			exit
		
		landmarks = landmarks.reshape(-1, 5)
		
		sample = {'image': image, 'landmarks': landmarks}
		#print(idx, fname, image.size, image.shape, type(image))
		print(COLOR.Green,'ID\t: {} in {}'.format(idx, len(self.fname)))
		print('Fname\t: {}'.format(fname))
		print('Shape\t: {}'.format(image.shape), COLOR.END)
		print(sample['landmarks'][0][0], sample['landmarks'][0][1], sample['landmarks'][0][2], sample['landmarks'][0][3], sample['landmarks'][0][3])
		
		return sample

	def imageCopy2Dest(self, sample, i):
		def  new_image_name(sample, i):
			name = sample['landmarks'][0][0]
			hasWater = sample['landmarks'][0][1]
			TimeEvent = sample['landmarks'][0][2]
			lat = sample['landmarks'][0][3]
			lon = sample['landmarks'][0][4]
			#return f'{name}_{hasWater}_{TimeEvent}_{lat}_{lon}.png'
			return f'{name}_{i:04n}_{hasWater}.png'

		def new_image_crop(image):
			"ATTENTION - it is important to crop Cutyscape dataset" 
			def center(image):
				w, h = image.size
				cw = int(w/2)
				ch = int(h/2)
				return cw, ch


			
			cw, ch = center(image)
			left = cw - int(1024/2)
			top = ch -int(768/2)-128
			right = cw + int(1024/2)
			bottom = ch +int(768/2)-128
			print(image.size)
			#left = 0 
			#upper = 0 
			#right = w
			#lower = bottom
			#left, upper, right, lower
			image = image.crop((left, top, right, bottom))
			image = image.resize((640, 480))
			return image

		image = Image.fromarray(sample['image'])
		image = new_image_crop(image)
		image.save(os.path.join(self.dest_dir, new_image_name(sample, i)))

def read_csv(i, csv_file):
	landmarks_frame = pd.read_csv(csv_file)
	img_name = landmarks_frame.iloc[i,0]
	img_id = landmarks_frame.iloc[i,1]
	img_TimeEvent = landmarks_frame.iloc[i,2]
	img_hasWater = landmarks_frame.iloc[i,3]
	img_lat, img_lon = landmarks_frame.iloc[i,4], landmarks_frame.iloc[i,5]

	return img_name, img_TimeEvent, img_id, img_hasWater, img_lat, img_lon


class DATA_PLOT:
	def __init__(self, path, csv_file):
		self.valid_img = ["png"]
		self.fname = []
		# frames with a water 
		self.fname_y = []
		# frames with no water 
		self.fname_n = []
		self.i_y = []
		self.i_n = []
		self.files2list(path, csv_file)

	def files2list(self, path, csv_file):
		self.path = path
		for i, fname in enumerate(os.listdir(path)):
			fname = os.path.join(path, fname)
			_, _, _, img_hasWater, _, _ = read_csv(i, csv_file)
			r = f'Has Water: {img_hasWater}'
			#print(i, fname, r)
			if img_hasWater==1: 
				self.fname_y.append(fname)
				self.i_y.append(i)
			else:
				self.fname_n.append(fname)
				self.i_n.append(i)
		print('Frames with no water: {}'.format(len(self.fname_n)))
		print()
		print('Frames with    water: {}'.format(len(self.fname_y)))

	def random_selection_train():
		
		pass



	def random_plot(self, csv_file, plot = True):
		def metadata(fname):
			frame = Image.open(fname)
			#exifdata = frame.getexif()
			#for tag_id in exifdata:
			#    tag = TAGS.get(tag_id, tag_id)
			#    data = exifdata.get(tag_id)
				#if isinstance(data, bytes):
				#    data = data.decode()
				#print(f"{tag:25}: {data}")

			return frame

		if plot:
			print( len(self.fname))                
			fig = plt.figure(str(len(self.fname)) + " images found in" + self.path, dpi=80, figsize=(18, 10)) #
			water = 0
			import random
			print(len(self.i_n), len(self.i_y))
			if len(self.i_n) ==0 and len(self.i_y) !=0:
				for i in range(3):
					ax = fig.add_subplot(1, 3, i + 1)
					r = self.i_y[random.randint(0,len(self.i_y)-1)]
					img_name, _, img_id, img_hasWater, _, _ = read_csv(r,csv_file)
					fname = os.path.join(self.path,f'{img_name}_{img_id:04d}_{img_hasWater}.png')
					frame = metadata(fname=fname)
					width, height = frame.size
					s = f'ID: {img_id}, Has Water: {img_hasWater}, W: {width}, H: {height}'
					ax.imshow(frame)
					r = f'ID: {img_id}, Has Water: {img_hasWater}, w: {width}, h {height}'
					ax.set_title(s, fontsize=11, color='red')
					plt.tight_layout()
					plt.subplots_adjust(wspace=0, hspace=0)				
					water=water+1

			if len(self.i_n) !=0 and len(self.i_y) ==0:
				for i in range(3):
					ax = fig.add_subplot(1, 3, i + 1)
					r = self.i_n[random.randint(0,len(self.i_n)-1)]
					img_name, _, img_id, img_hasWater, _, _ = read_csv(r,csv_file)
					fname = os.path.join(self.path,f'{img_name}_{img_id:04d}_{img_hasWater}.png')
					frame = metadata(fname=fname)
					width, height = frame.size
					s = f'ID: {img_id}, Has Water: {img_hasWater}, W: {width}, H: {height}'
					ax.imshow(frame)
					r = f'ID: {img_id}, Has Water: {img_hasWater}, w: {width}, h {height}'
					ax.set_title(s, fontsize=11, color='green')
					plt.tight_layout()
					plt.subplots_adjust(wspace=0, hspace=0)				
					water=water+1

			if len(self.i_n) !=0 and len(self.i_y) !=0:
			
				for i in range(6):
					ax = fig.add_subplot(2, 3, i + 1)
					if water <3:
						r = self.i_y[random.randint(0,len(self.i_y)-1)]
						img_name, _, img_id, img_hasWater, _, _ = read_csv(r,csv_file)
					else:
						r = self.i_n[random.randint(0,len(self.i_n)-1)]
						img_name, _, img_id, img_hasWater, _, _ = read_csv(r,csv_file)


					
				
					fname = os.path.join(self.path,f'{img_name}_{img_id:04d}_{img_hasWater}.png')
					frame = metadata(fname=fname)
					width, height = frame.size
					s = f'ID: {img_id}, Has Water: {img_hasWater}, W: {width}, H: {height}'
					ax.imshow(frame)
					r = f'ID: {img_id}, Has Water: {img_hasWater}, w: {width}, h {height}'
					if water <3:
						ax.set_title(s, fontsize=11, color='red')
					else:
						ax.set_title(s, fontsize=11, color='green')
				#	ax.set_aspect('equal')
					plt.tight_layout()
					plt.subplots_adjust(wspace=0, hspace=0)				
					water=water+1
			plt.show()
			#fig.savefig("desktop.pdf")

def random_selection(source_img, sourse_json, dest, csv_name, no_of_files):
	import os
	import random
	import shutil
	import json
	from PIL import Image
	import csv
	"""
	source "Enter the Source Directory
	dest Enter the Destination Directory
	no_of_files Enter The Number of Files To Select
	"""
	# Using for loop to randomly choose multiple files
	m = 0
	dict_CSV = {}
	img_nm = []
	img_id_ = []
	img_TiE = []
	img_lat = []
	img_lon = []
	img_hW_ = []

	for i in range(no_of_files):
		# Variable random_file stores the name of the random file chosen
		random_file = random.choice(os.listdir(source_img))

		name, extension = os.path.splitext(random_file)
		city = name.split('_')[0]
		#print('city', city)
		num0 = name.split('_')[1]
		num1 = name.split('_')[2]

		random_file = os.path.join(source_img, random_file)
		check_if_file_existed(random_file)
		json_file = os.path.join(
			sourse_json, f'{city}_{num0}_{num1}_vehicle.json')
		check_if_file_existed(json_file)

		exif = Image.open(random_file).getexif()
		creation_time = exif.get(36867)
		with open(json_file, 'r') as jsonfile:
			data = jsonfile.read()
		# parse file
		obj = json.loads(data)
		t = "1" + num0 + num1

		lat = format(float(obj["gpsLatitude"]), ".4f")
		lon = format(float(obj["gpsLongitude"]), ".4f")
		hw = 0

		new_img_name = os.path.join(dest, f'{city}_{i:04n}_{hw}.png')
		img_nm.append(city)
		img_id_.append(f'{i:04n}')
		img_TiE.append(currentTime2Millisec())
		img_lat.append(str(format(float(obj["gpsLatitude"]), ".4f")))
		img_lon.append(str(format(float(obj["gpsLongitude"]), ".4f")))
		img_hW_.append(hw)
		print(city, t, hw, format(float(obj["gpsLatitude"]), ".4f"), format(
			float(obj["gpsLongitude"]), ".4f"))
		m = m + 1
		# print(str(obj["gpsLatitude"]))
		# print(str(obj["gpsLongitude"]))
		# print("%d} %s"%(i+1,check_if_dir_existed(os.path.join(source_img, random_file))))
		# source_file="%s\%s"%(source,random_file)
		# dest_file=dest
		# "shutil.move" function moves file from one directory to another
		shutil.move(random_file, new_img_name)

		print(random_file)
		print(source_img)
		# os.rename(os.path.join(dest, random_file), new_img_name)
		# os.rename(os.path.join(dest, random_file), os.path.join(dest, new_img_name))

	dict_CSV["dataset"] = img_nm
	dict_CSV['image_id'] = img_id_
	dict_CSV['TimeEvent'] = img_TiE
	dict_CSV['hasWater'] = img_hW_
	dict_CSV['lat'] = img_lat
	dict_CSV['lon'] = img_lon


	print("\n\n" + "$" * 33 + "[ Files Moved Successfully ]" + "$" * 33)
	with open(csv_name, 'w', newline="") as csv_file:
		w = csv.writer(csv_file, delimiter=',')
		w.writerow(dict_CSV.keys())
		w.writerows(zip(*dict_CSV.values()))
		csv_file.close()
	#print(dict_CSV)
	print('csv done')
