import argparse
import os
from utils import load_img, imshow

parser = argparse.ArgumentParser(description='Enter content image and styling image')

parser.add_argument('--content',type=str,required=True,
                    help='Content Image Path')

parser.add_argument('--style',type=str,required=True,
                    help='Style Image Path')
args = parser.parse_args()

data_dir = './data'
content_dir = os.path.join(data_dir,args.content)
style_dir = os.path.join(data_dir,args.style)

content_image = load_img(content_dir)
style_image = load_img(style_dir)

imshow(content_image)
imshow(style_image)
