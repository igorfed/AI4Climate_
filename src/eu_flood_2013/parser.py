import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from com.common import *
import json
from PIL import Image
import matplotlib.pyplot as plt

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
def arg_parser():
    parser = argparse.ArgumentParser(description = 'This is a program to pars and annotate eu-flood-2013-small images')
    parser.add_argument('-source', '--source', required=False, type=str, help='path to the folder with dataset')
    parser.add_argument('-flooding', '--flooding', required=False, type=str, help='path to flooding.txt')
    #parser.add_argument('-irrelevant', '--irrelevant', required=False, type=str, help='path to irrelevant.txt')
    #parser.add_argument('-metadata', '--metadata', required=False, type=str, help='path to metadata.json')
    parser.add_argument('-out', '--out', required=False, type=str, help='ath to the output folder')
    return parser.parse_args()

def getCurrentTime():
	from datetime import datetime
	now = datetime.now()
	return now.strftime("%Y_%m_%d_%H_%M_%S")

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
        for i, fname in enumerate(os.listdir(self.source_dir)):
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
            print(f'fname{name}File not accessible')
        finally:
            f.close()

        #image = io.imread(fname)
        
        image = Image.open(fname).convert('RGB')
        image.verify()
		
        name = os.path.basename(name)
        #print(idx,name)
        sample = {'image': image, 'name': name}
        return sample

        

def specific_folder(out, f='out'):
    out = os.path.join(out, f)
    if not os.path.exists(out):
        print(COLOR.Red + f'Output folder is not available' + COLOR.END)
        os.makedirs(out)
        print(COLOR.Green + f'Output folder created in : {out}' + COLOR.END)
        if os.path.exists(out):
            return (True, out)
    return (False, out)

def default_folder(out='out'):
    if not os.path.exists(out):
        print(COLOR.Red + f'Output folder is not available' + COLOR.END)
        os.makedirs(out)
        print(COLOR.Green + f'Output folder created in : {(sys.path[1])}' + COLOR.END)
        if os.path.exists(out):
            return (True, out)
    return (False, out)
    
def write_csv(out, dataset, dict_CSV):
    csv_name = os.path.join(out, dataset+'.csv')
    with open(csv_name, 'w', newline="") as csv_file:
        w = csv.writer(csv_file, delimiter=',')
        w.writerow(dict_CSV.keys())
        w.writerows(zip(*dict_CSV.values()))
        csv_file.close()
        print("CSV Done", csv_name)
    return csv_name


def main():
    args = arg_parser()
    check_if_dir_existed(args.source)
    check_if_file_existed(args.flooding)
    #check_if_file_existed(args.irrelevant)
    #check_if_file_existed(args.metadata)
    __dataset = EU2013(source = args.source, out = args.out)
    flooding = __dataset.readTextFile(filename=args.flooding)
    #irrelevant = __dataset.readTextFile(filename=args.irrelevant)
    dataset= 'eu2013'

    if args.out == None:
        _, out = default_folder(out='out')
    else:
        _, out= specific_folder(out=args.out, f='out')


    dict_CSV = {}
    img_nm = []
    img_id_ = []
    hasWater = []
    img_TiE = []
    img_lat = []
    img_lon = []
    i = 0
    #for idx in range(len(__dataset)):
    for idx in range(len(__dataset)):
        sample = __dataset[idx]
        img = sample['image']
        #print(sample['name'])
        if (sample['name'] in flooding):
        #if (sample['name'] in flooding) or (sample['name'] in irrelevant):
            img_nm.append(dataset)
            img_id_.append(f'{i:04n}')
            img_TiE.append(currentTime2Millisec())
            img_lat.append('12.3456')
            img_lon.append('78.9012')
            #-----------------------------#
            hasWater.append(1)
            _, path2parsedImage= specific_folder(out=out, f=str(hasWater[-1]))
            new_fname = f'{dataset}_{i:04n}_{hasWater[-1]}.png'
            #new_fname = str(sample['name']) + '.png'
            sample['name']
            print(COLOR.Green + str(i), str(idx), new_fname +COLOR.END)
            img.save(os.path.join(path2parsedImage, new_fname))
            i=i+1

    dict_CSV["dataset"] = img_nm
    dict_CSV["image_id"] = img_id_
    dict_CSV["TimeEvent"] = img_TiE
    dict_CSV["lan"] = img_lat
    dict_CSV["lon"] = img_lon
    dict_CSV["hasWater"] = hasWater
    csv_name = write_csv(out, dataset, dict_CSV)
    print('Number of flooded images in the dataset\t: {}'.format(len(flooding)))
    __ai4ci = DATA_PLOT()
    print(f'image_folder{path2parsedImage}')
    __ai4ci.files2list(path=path2parsedImage)
    __ai4ci.random_plot(csv_file=os.path.join(out, csv_name), plot=True)


if __name__ == '__main__':
    main()
    

    print('done')


