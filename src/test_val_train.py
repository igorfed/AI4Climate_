import csv
import os
from tabnanny import check
import pandas as pd
from com.common_packages import getCurrentTime, check_if_dir_existed, check_if_file_existed
import random
from time import time
import sys
from pathlib import Path
import numpy as np
from com.colors import COLOR
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse




def destination_path(dest):
	'''
	Check if output directory exist 
	'''
	from pathlib import Path
	if dest != None:
		'return specified directory'
		return dest
	else:
		'return default directory'
		return os.path.join((Path(__file__).parent.parent), 'temp_' + time)

def train_images(root_dir, annotations, percentage=0.8):
    def random_n_file_selection(annotations):
        for i in range(len(annotations)):
            pass
    N = int(len(annotations)*percentage)
    print('percentage :', percentage, '= ', N)
    train_list = range(0,N-1)
    train_list =  random.sample(train_list, k = int(len(annotations)))


def bubbleSort(arr):
    n = len(arr)
    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    # Traverse through all array elements
    for i in range(n-1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
         
        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return


class DATA_to_TEST_VAL_TRAIN:
	
    def __init__(self, images_folder, csv_file, dest, percentage, dataset_num):

    #def __init__(self, path, csv_file, dest, percentage=0.8, dataset_num=4):
        '''path - path to the images and csv'
        '''
        #self.path = path
        self.valid_img = ["png"]
        self.fname = []
        # frames with a water 
        self.fname_y = []
		# frames with no water 
        self.fname_n = []
        self.i_y = []
        self.i_n = []
        self.dest = dest
        self.percentage = percentage
        # full path to csv
        self.csv_file = os.path.join(images_folder, csv_file)
        self.df = pd.read_csv(self.csv_file)
        
        self.image_path =  images_folder
        self.length_of_data()
        self.dest_test = os.path.join(self.dest, 'test')
        
        self.dest_valid = os.path.join(self.dest, 'valid')
        
        self.dest_train = os.path.join(self.dest, 'train')
        self.dataset_num = dataset_num
        check_if_dir_existed(self.dest_test,True)
        check_if_dir_existed(self.dest_valid,True)
        check_if_dir_existed(self.dest_train,True)

        self.files2list()


    def length_of_data(self):

        N = len(self.df)
        #N = 100
        list_total = list(range(0,N))
        

        def in_list(list1,list2):
            list_diff = []
            for i in list1:
                if i not in list2:
                    list_diff.append(i)
            return list_diff

        def check_sum(A, B, C, X):
            if (A, B, C == len(X)):
                print('True')
            else:
                print('False')            
        
        def check_list(A, B, C, X):
            if (len(A) + len(B) + len(C) == len(X)):
                print('True')
            else:
                print('False')            
        

        def check_dublicated_values(lst):
            set_lst = set(lst)
            contains_duplicates = len(lst) != len(set_lst)
            print('dublicates ', contains_duplicates)

        length_train = int(len(list_total)*self.percentage)
        length_valid = int((len(list_total) - length_train)/2)
        length_test = int(len(list_total) - length_train - length_valid)
        check_sum(length_test, length_train, length_valid, list_total)
        
        
        #print(f'Total images {len(N)} \t{self.percentage*100} %')
        #print(f'Total images {len(N)} \t{self.percentage*100} %')
        print('train :\t', length_train, ', percentage :\t', self.percentage)
        print('val   :\t', length_valid)
        print('test  :\t', length_test)


        
        
        list_train =  random.sample(list_total, k = length_train)
        bubbleSort(list_train)
        list_valid_test = in_list(list_total, list_train)
        check_dublicated_values(list_train)
        ###################################
        #print('list_train', len(list_train) )
        
        list_valid =  random.sample(list_valid_test, k = length_valid)
        bubbleSort(list_valid)
        #print('list_valid', len(list_valid) )

        list_test = in_list( list_valid_test, list_valid)
        #print('list_test', len(list_test))
        #check_list(list_train, list_valid, list_test, list_total)
        self.list_train = list_train
        self.list_test = list_test
        self.list_valid = list_valid

        #print(len(list_valid), np.min(list_valid), np.max(list_valid), list_valid )
        
        #list_valid =  random.sample(list_valid_test, k = length_valid)
        #print(len(list_valid))
        #list_test = in_list(list_valid_test, list_valid )
        #print(len(list_test))

        #if (len(list_test) + len(list_valid) + len(list_test) == len(self.df)):
        #    print('True')
        #else:
        #    print('False')            
        

    


    def files2list(self):
        ################################
        def read_csv(i, df):
            img_fname = df.iloc[i,0] 
            img_name = df.iloc[i,1]
            img_hasWater = df.iloc[i,2]
            
            return img_fname, img_name, img_hasWater

        # Total -> test -> (Class0 Class1)
        ################################        
        def check_img(fname):
            try:
                f = open(fname)
            except IOError:
                print(COLOR.Red + f'fname{fname}File not accessible' + COLOR.END)
            finally:
                f.close()
        
        def classes(i, img_hasWater, list, dest, image_name):
            if i in list:
                if img_hasWater==0: 
                    label_path = os.path.join(dest, 'Class0')
                    check_if_dir_existed(label_path, True)
                
                if img_hasWater==1: 
                    label_path = os.path.join(dest, 'Class1')
                    check_if_dir_existed(label_path, True)
                print(label_path, image_name)
                image.save(os.path.join(label_path, image_name))    

        for i, fname in enumerate(os.listdir(self.image_path)):
            img_fname, img_name, img_hasWater = read_csv(i, self.df)
            #if self.dataset_num ==4:
            r = os.path.join(self.image_path, img_fname)
            image = Image.open(r).convert('RGB')
            image.verify()

            if self.dataset_num ==4:
                if img_name in ['desktop', 'mobile', 'roadway', 'eu2013', 'berlin', 'munich', 'mainz', 'zurich']:
                    print(i, img_fname, img_name, type(self.dataset_num), img_hasWater)
                    image = Image.open(r).convert('RGB')
                    image.verify()
                    classes(i, img_hasWater, self.list_train, self.dest_train, img_fname)
                    classes(i, img_hasWater, self.list_test, self.dest_test, img_fname)
                    classes(i, img_hasWater, self.list_valid, self.dest_valid, img_fname)
            elif self.dataset_num ==3:
                if img_name in ['desktop', 'mobile', 'roadway', 'berlin', 'munich', 'mainz', 'zurich']:
                    print(i, img_fname, img_name, type(self.dataset_num), img_hasWater)
                    image = Image.open(r).convert('RGB')
                    image.verify()
                    classes(i, img_hasWater, self.list_train, self.dest_train, img_fname)
                    classes(i, img_hasWater, self.list_test, self.dest_test, img_fname)
                    classes(i, img_hasWater, self.list_valid, self.dest_valid, img_fname)
            elif self.dataset_num ==2:
                if img_name in ['desktop', 'mobile', 'berlin', 'munich', 'mainz', 'zurich']:
                    print(i, img_fname, img_name, type(self.dataset_num), img_hasWater)
                    image = Image.open(r).convert('RGB')
                    image.verify()
                    classes(i, img_hasWater, self.list_train, self.dest_train, img_fname)
                    classes(i, img_hasWater, self.list_test, self.dest_test, img_fname)
                    classes(i, img_hasWater, self.list_valid, self.dest_valid, img_fname)
            elif self.dataset_num ==1:
                if img_name in ['desktop', 'mobile']:
                    print(i, img_fname, img_name, type(self.dataset_num), img_hasWater)
                    image = Image.open(r).convert('RGB')
                    image.verify()
                    classes(i, img_hasWater, self.list_train, self.dest_train, img_fname)
                    classes(i, img_hasWater, self.list_test, self.dest_test, img_fname)
                    classes(i, img_hasWater, self.list_valid, self.dest_valid, img_fname)
                



def arg_parser():
    parser = argparse.ArgumentParser(description = 'This is a random selection of images program')
    parser.add_argument('-source', '--source', required=False, type=str, help='Source of images')
    parser.add_argument('-csv', '--csv', required=False, type=str, help='Name of annotations file')
    parser.add_argument('-dest', '--dest', required=False, type=str, help='Destination to copy')
    parser.add_argument('-type', '--type', required=False, type=int, help='Type of the dataset [mobile, desktop, roadway, eu2013 ...]')
    parser.add_argument('-p', '--p', required=False, type=str, help='Percaintage from the sourse')
    return parser.parse_args()



if __name__ == '__main__':

    #root_dir = '/media/igofed/SSD_1T/AI4CI/FULLDATASET/FULLDATASET'
    #csv_file = "annotation.csv"
    
    args = arg_parser()
    images_folder = os.path.join(args.source, 'image')
    check_if_dir_existed(images_folder)
    csv_file = os.path.join(args.source, args.csv)
    check_if_file_existed(csv_file)
    dest = os.path.join((Path(__file__).parent.parent), args.dest)
    check_if_dir_existed(dest,True)
    __train = DATA_to_TEST_VAL_TRAIN(images_folder, csv_file, dest, percentage=0.7, dataset_num = args.type)

    

    print('done')