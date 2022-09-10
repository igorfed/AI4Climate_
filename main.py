'''
@author: Igor F
'''

import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Top level of Ai4Climate')
    parser.add_argument('--method_name', help='pass method name to be used', choices=adict)
    return parser

def euflood():
    return ("euflood")

def roadmap():
    return ("roadmap")

def desktop():
    return ("desktop")

def mobile():
    return ("mobile")


adict = {   'euflood': euflood, 
            'roadmap': roadmap,
            'roadmap': desktop,
            'mobile': mobile}  # map names to functions

def where_are_you(method_name):
    fn = adict[method_name]
    # could also use getattr
    return fn()

def main():
    args = create_parser().parse_args()     # get the string
    print(args.method_name)                 # debugging
    return where_are_you(args.method_name)

if __name__ == '__main__':
    print(main()) 
    print('done')