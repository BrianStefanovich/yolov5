import torch
import argparse
import os
import sys
from pathlib import Path
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def main(
        dataroot="",  # show results
        path_to_yolov5="",  # save results to *.txt
        path_to_weigths="",  # save confidences in --save-txt labels
        result_dir="",  # save cropped prediction boxes
        save_image=False  # save cropped prediction boxes
        ):

    # Model
    model = torch.hub.load(path_to_yolov5,'custom', path=path_to_weigths, source='local')

    # Image
    dataroot = os.listdir(dataroot)

    img_name = dataroot[0]
    im = Image.open(os.path.join(dataroot,img_name))

    # Inference
    results = model(im)
    box = results.xyxy[0] #Escupe un array de (x,y,x,y) con todas las ocurrencias

    #Crop
    cropped_im = im.crop(box)

    if(save_image):
        cropped_im.save(os.path.join(result_dir,img_name))
        return

    cropped_im.save(sys.stdout, 'PNG')
    return

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default=ROOT/'data/images', help='Path to image')
    parser.add_argument('--path-to-yolov5', type=str, default=ROOT/'yolov5', help='Path to root yolo directory')
    parser.add_argument('--path-to-weigths', type=str, default=ROOT/'yolov5/runs/weights/best.pt', help='Path to weights file')
    parser.add_argument('--result-dir', type=str, default=ROOT/'data/images', help='Path to store cropped image')
    parser.add_argument('--save-image', action='store_true', help='Save cropped image')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))