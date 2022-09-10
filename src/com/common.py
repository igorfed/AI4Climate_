import numpy as np
import math
import os
from PIL import Image
from skimage import io, transform
from datetime import datetime
import csv
import pandas as pd
import matplotlib.pyplot as plt

class COLOR:
   Black = "\[\033[0;30m\]"  # Black
   Red = "\033[0;31m"  # Red
   Green = "\033[0;32m"  # Green
   Yellow = "\[\033[0;33m\]"  # Yellow
   Blue = "\033[0;34m"  # Blue
   Purple = "\[\033[0;35m\]"  # Purple
   Cyan = "\[\033[0;36m\]"  # Cyan
   White = "\[\033[0;37m\]"  # White

   # Bold
   BBlack = "\[\033[1;30m\]"  # Black
   BRed = "\033[1;31m"  # Red
   BGreen = "\033[1;32m"  # Green
   BYellow = "\[\033[1;33m\]"  # Yellow
   BBlue = "\033[1;34m"  # Blue
   BPurple = "\[\033[1;35m\]"  # Purple
   BCyan = "\[\033[1;36m\]"  # Cyan
   BWhite = "\[\033[1;37m\]"  # White

   # Underline
   UBlack = "\[\033[4;30m\]"  # Black
   URed = "\[\033[4;31m\]"  # Red
   UGreen = "\[\033[4;32m\]"  # Green
   UYellow = "\[\033[4;33m\]"  # Yellow
   UBlue = "\[\033[4;34m\]"  # Blue
   UPurple = "\[\033[4;35m\]"  # Purple
   UCyan = "\[\033[4;36m\]"  # Cyan
   UWhite = "\[\033[4;37m\]"  # White

   # Background
   On_Black = "\[\033[40m\]"  # Black
   On_Red = "\[\033[41m\]"  # Red
   On_Green = "\[\033[42m\]"  # Green
   On_Yellow = "\[\033[43m\]"  # Yellow
   On_Blue = "\[\033[44m\]"  # Blue
   On_Purple = "\[\033[45m\]"  # Purple
   On_Cyan = "\[\033[46m\]"  # Cyan
   On_White = "\[\033[47m\]"  # White

   # High Intensty
   IBlack = "\[\033[0;90m\]"  # Black
   IRed = "\[\033[0;91m\]"  # Red
   IGreen = "\[\033[0;92m\]"  # Green
   IYellow = "\[\033[0;93m\]"  # Yellow
   IBlue = "\[\033[0;94m\]"  # Blue
   IPurple = "\[\033[0;95m\]"  # Purple
   ICyan = "\[\033[0;96m\]"  # Cyan
   IWhite = "\[\033[0;97m\]"  # White

   # Bold High Intensty
   BIBlack = "\[\033[1;90m\]"  # Black
   BIRed = "\[\033[1;91m\]"  # Red
   BIGreen = "\[\033[1;92m\]"  # Green
   BIYellow = "\[\033[1;93m\]"  # Yellow
   BIBlue = "\033[1;94m"  # Blue
   BIPurple = "\[\033[1;95m\]"  # Purple
   BICyan = "\[\033[1;96m\]"  # Cyan
   BIWhite = "\[\033[1;97m\]"  # White

   # High Intensty backgrounds
   On_IBlack = "\[\033[0;100m\]"  # Black
   On_IRed = "\[\033[0;101m\]"  # Red
   On_IGreen = "\[\033[0;102m\]"  # Green
   On_IYellow = "\[\033[0;103m\]"  # Yellow
   On_IBlue = "\[\033[0;104m\]"  # Blue
   On_IPurple = "\[\033[10;95m\]"  # Purple
   On_ICyan = "\[\033[0;106m\]"  # Cyan
   On_IWhite = "\[\033[0;107m\]"  # White
   END = '\033[0m'


def check_if_file_existed(filename):
	if os.path.isfile(filename):
		print(COLOR.Blue + f'filename \t: {filename} existed' + COLOR.END)
	else:
		print(COLOR.Red + f'filename \t: {filename} is not existed' + COLOR.END)


def check_if_dir_existed(dir_name):

	if not os.path.exists(dir_name):
		print(COLOR.Red +
			  f'folder \t\t: {dir_name} is not available' + COLOR.END)
	else:
		print(COLOR.BBlue + f'folder \t\t: {dir_name} is available' + COLOR.END)


def check_if_dir_existed_create(dir_name):

	if not os.path.exists(dir_name):
		print(COLOR.Red +
			  f'folder \t\t: {dir_name} is not available' + COLOR.END)
		os.mkdir(dir_name)
		print(COLOR.Green + f'folder \t\t: {dir_name} created' + COLOR.END)
	else:
		print(COLOR.Blue + f'folder \t\t: {dir_name} is available' + COLOR.END)


def min(x):
   return round(np.min(x), 3)


def max(x):
   return np.around(np.max(x), 3)


def mean(x):
   return np.around(np.mean(x), 3)


def std(x):
   return np.around(np.std(x), 3)


def getCurrentTime():
	from datetime import datetime
	now = datetime.now()
	return now.strftime("%Y_%m_%d_%H_%M_%S")

def currentTime2Millisec():
	'''
	Getting the current time in format YYYY MM DD HH MM SS
	and transfrom it into millisec
	'''
	curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
	dt_obj = datetime.strptime(str(curr_datetime), '%Y-%m-%d %H-%M-%S')
	millisec = dt_obj.timestamp()*1000
	return int(millisec)

def random_selection(source_img, sourse_json, dest, no_of_files):
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
	csv_name = os.path.join(dest, city + '.csv')
	with open(csv_name, 'w', newline="") as csv_file:
		w = csv.writer(csv_file, delimiter=',')
		w.writerow(dict_CSV.keys())
		w.writerows(zip(*dict_CSV.values()))
		csv_file.close()
	#print(dict_CSV)
	print('csv done')


class DESKTOP():

	def __init__(self, source, dest):
		t = COLOR.Green + 80 * '=' + COLOR.END
		self.valid_img = [".jpg", "jpeg"]
		print(t)
		"""
		root_dir is the path to dataset
		output is the output dir for renamed images
		"""
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

		fname = self.fname[idx]
		name, extension =  os.path.splitext(fname)
		try:
			f = open(fname)
		except IOError:
			print(f'fname{fname}File not accessible')
		finally:
			f.close()
		image = io.imread(fname)
		print(idx,fname)
		landmarks = np.array(image_name(os.path.basename(name)))
		landmarks = landmarks.reshape(-1, 5)
		sample = {'image': image, 'landmarks': landmarks}
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
		image = Image.fromarray(sample['image'])
		image.save(os.path.join(self.dest_dir, new_image_name(sample, i)))


		#self.frames name.split('_')[0]
	#			mobile = name.split('_')[0]
	#			hw = name.split('_')[1]
	#			t = name.split('_')[2]
	#			lat = format(float(name.split('_')[3]), ".4f")
	#			lon = format(float(name.split('_')[4]),".4f")
	#			#new_fname = f'{i:04n}_{mobile}_{hw}_{t}_{lat}_{lon}.png'
	#			new_fname = f'{mobile}_{hw}_{t}_{lat}_{lon}.png'
	#			im.save(os.path.join(self.path2dataset, new_fname))
	#			print(new_fname)
	#			i = i +1
	#			im.close()



	def writeCSV(self):
		import csv


def extract_exif(path, csv_name):
	i = 0
	id = []
	img_nm = []
	dateTimeEvent = []
	LAT  = []
	LON = []
	hasWater = []
	dict_CSV = {}
	roadway = 'roadway'
	for f in os.listdir(path):
		name , extension = os.path.splitext(f)
		img = Image.open(os.path.join(path,f)).convert('RGB')
		img.verify()
		stat = os.stat(os.path.join(path,f))
		t = int(stat.st_mtime)*1000 + i
		lat = '12.3456'
		lon = '78.9012'
		new_fname = f'{roadway}_{i:04n}_{1}.png'

		#img.close()

		img_nm.append(roadway)
		id.append(f'{i:04n}')
		dateTimeEvent.append(currentTime2Millisec())
		LAT.append(lat)
		LON.append(lon)
		hasWater.append(1)
		img.save(os.path.join(path, new_fname))
		i = i+1
		print(os.path.join(path, new_fname))

	dict_CSV["dataset"] = img_nm
	dict_CSV["image_id"] = id
	dict_CSV["TimeEvent"] = dateTimeEvent
	dict_CSV["lan"] = LAT
	dict_CSV["lon"] = LON
	dict_CSV["hasWater"] = hasWater


	with open(csv_name, 'w', newline="") as csv_file:
		w = csv.writer(csv_file, delimiter=',')
		w.writerow(dict_CSV.keys())
		w.writerows(zip(*dict_CSV.values()))
		csv_file.close()
		print("CSV Done", csv_name)

#path= "/media/igofed/SSD_1T/AI4CI/Carlo/roadway/Dataset/images/"
#csv_name= "/media/igofed/SSD_1T/AI4CI/Carlo/roadway/Dataset/roadway.csv"
#print(extract_exif(path, csv_name))


def read_csv(i, csv_file):
	landmarks_frame = pd.read_csv(csv_file)
	img_id = landmarks_frame.iloc[i,1]
	img_TimeEvent = landmarks_frame.iloc[i,2]
	img_lat, img_lon = landmarks_frame.iloc[i,3], landmarks_frame.iloc[i,4]
	img_hasWater = landmarks_frame.iloc[i,5]
	return img_TimeEvent, img_id, img_hasWater, img_lat, img_lon


class DATA_PLOT:
    def __init__(self):
        self.valid_img = [".jpg", "png"]
        self.fname = []

    def files2list(self, path):
        self.path = path
        for i, fname in enumerate(os.listdir(path)):
            fname = os.path.join(path, fname)
            print(i, fname)
            #if os.path.splitext(fname)[-1].lower() not in self.valid_img:
            #    continue
            self.fname.append(fname)

        print(len(self.fname))

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
            fig = plt.figure(str(len(self.fname)) + " images found in" + self.path, figsize=(30, 30), dpi=80)
            for i in range(9):
                import random
                r = random.randint(0,len(self.fname))
                img_TimeEvent, img_id, img_hasWater, img_lat, img_lon = read_csv(r,csv_file)
                ax = fig.add_subplot(3, 3, i + 1)
                frame = metadata(fname=self.fname[r])
                head, tail = os.path.split(self.fname[r])
                width, height = frame.size
                #print(width, height)
                ax.imshow(frame)
                r = f'ID:{img_id}, Water:{img_hasWater}, w:{width}, h{height}'
                if img_hasWater:
                    ax.set_title(r, fontsize=12, color='red')
                else:
                    ax.set_title(r, fontsize=12, color='green')
                ax.set_aspect('equal')
            plt.show()
            #fig.savefig("desktop.pdf")