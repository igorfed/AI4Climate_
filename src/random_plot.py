import argparse
from importlib.resources import path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from com.common_packages import check_if_dir_existed
from com.common_packages import check_if_file_existed
from com.common_packages import DATA_PLOT
import json
from PIL import Image
import matplotlib.pyplot as plt
'''
This is a program to plot random images and read annotated csv to random check the images
Args: source - path to the folder with dataset
    : flooding - path to the text file with flooding
    : csv - path to the text file with irrelevant images for non flooding
'''

def arg_parser():
    parser = argparse.ArgumentParser(description = 'This is a program to plot random images and read annotated csv to random check of the images')
    parser.add_argument('-source', '--source', required=False, type=str, help='path to the folder with dataset')
    parser.add_argument('-csv', '--csv', required=False, type=str, help='path to csv file')
    return parser.parse_args()


def main():
    args = arg_parser()
    check_if_dir_existed(args.source)
    check_if_file_existed(args.csv)
    __ai4ci = DATA_PLOT(path=args.source, csv_file=args.csv)
    print(f'image_folder{args.source}')
    __ai4ci.random_plot(csv_file=args.csv, plot=True)


if __name__ == '__main__':
    main()
    print('done')
