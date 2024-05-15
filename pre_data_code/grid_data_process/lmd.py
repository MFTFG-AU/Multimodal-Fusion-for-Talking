import cv2
import numpy as np
import face_alignment
from skimage import io
import torch
import torch.nn.functional as F
import json
import os
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import argparse


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                     device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32,
                       device=euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,
                    default='Obama_', help='identity of target person')
parser.add_argument('--step', type=int,
                    default=0, help='step for running')

args = parser.parse_args()
id = args.id
vid_file = os.path.join('/fs1/home/tjuvis_2022/lxx/AD_nerf/dataset/', 'vids', id + '.mp4')
running_step = args.step
if running_step in [0, 1]:
    if not os.path.isfile(vid_file):
        print('no video')
        # exit()

id_dir = os.path.join('dataset', id)
# Path(id_dir).mkdir(parents=True, exist_ok=True)
# id_dir = os.path.join('dataset', id, '0')
Path(id_dir).mkdir(parents=True, exist_ok=True)
ori_imgs_dir = os.path.join(id_dir, 'ori_imgs')
Path(ori_imgs_dir).mkdir(parents=True, exist_ok=True)
parsing_dir = os.path.join(id_dir, 'parsing')
Path(parsing_dir).mkdir(parents=True, exist_ok=True)
head_imgs_dir = os.path.join(id_dir, 'head_imgs')
Path(head_imgs_dir).mkdir(parents=True, exist_ok=True)
com_imgs_dir = os.path.join(id_dir, 'com_imgs')
Path(com_imgs_dir).mkdir(parents=True, exist_ok=True)
torso_imgs = os.path.join(id_dir, 'torso_imgs')
Path(torso_imgs).mkdir(parents=True, exist_ok=True)

# whi_head_imgs_dir = os.path.join(id_dir, 'whi_head_imgs')
# Path(whi_head_imgs_dir).mkdir(parents=True, exist_ok=True)


# running_step = args.step
running_step = 2
# # Step 0: extract wav & deepspeech feature, better run in terminal to parallel with
# below commands since this may take a few minutes
if running_step == 0:
    print('--- Step0: extract deepspeech feature ---')
    wav_file = os.path.join(id_dir, 'aud.wav')
    # extract_wav_cmd = 'ffmpeg -i ' + vid_file + ' -f wav -ar 16000 ' + wav_file
    # os.system(extract_wav_cmd)
    extract_ds_cmd = 'python /fs1/home/tjuvis_2022/lxx/DFRF-main/data_util/deepspeech_features/extract_ds_features.py --input=' + id_dir
    os.system(extract_ds_cmd)
    exit()

# Step 1: extract images
if running_step == 1:
    print('--- Step1: extract images from vids ---')
    cap = cv2.VideoCapture(vid_file)
    frame_num = 0
    while (True):
        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.resize(frame, (512, 512))
        cv2.imwrite(os.path.join(ori_imgs_dir, str(frame_num) + '.jpg'), frame)
        frame_num = frame_num + 1
    cap.release()
    running_step += 1
    # exit()


# Step 2: detect lands
if running_step == 2:
    print('--- Step 2: detect landmarks ---')
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False)
    for image_path0 in os.listdir('/fs1/home/tjuvis_2022/lxx/qikan/data/GRID_new_data/'):
        for image_path1 in os.listdir('/fs1/home/tjuvis_2022/lxx/qikan/data/GRID_new_data/'+image_path0):
            image_path2='/fs1/home/tjuvis_2022/lxx/qikan/data/GRID_new_data/'+image_path0+'/'+image_path1+'/'+'align_face/'
            for image_path in os.listdir(image_path2):
                if image_path.endswith('.jpg'):
                    input = io.imread(os.path.join(image_path2, image_path))[:, :, :3]
                    preds = fa.get_landmarks(input)
                    #print(image_path)
                    if len(preds) > 0:
                        lands = preds[0].reshape(-1, 2)[:,:2]
                        np.savetxt(os.path.join(image_path2, image_path[:-3] + 'lms'), lands, '%f')
    running_step+=1


