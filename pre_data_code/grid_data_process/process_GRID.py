import os
import numpy as np
import cv2
import sys
import dlib
import skimage.io as io
import scipy.io as sio
import subprocess
import shutil
from mfcc import wav_to_mfcc


main_path='D:/GRID/'
spearker_names_list=os.listdir(main_path)
spearker_names_list.sort()
save_path = 'D:/GRID_new_data/'
lm_path='D:/pro/landmark_68.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(lm_path)

#average landmarks
avglm1=([ 22,  63],[ 23,  90],[ 28, 117],[ 34, 143],[ 42, 169],[ 57, 191],[77, 208],[ 99, 222],[124, 226],[148, 220],[168, 206],[187, 188],[200, 166],
       [208, 140],[212, 114],[216,  87],[216,  60],[ 35,  51],[ 48,  37],[ 67,  33],[ 85,  36],[103,  41],[136,  40],[155,  34],[173,  31],[191,  34],
       [203,  47],[121,  59],[121,  77],[122,  94],[123, 113],[102, 125],[112, 129],[122, 131],[132, 128],[142, 125],[ 57,  64],[ 68,  56],[ 83,  56],
       [ 95,  65],[ 82,  69],[ 68,  69],[146,  64],[158,  54],[172,  54],[183,  62],[173,  67],[159,  67],[ 86, 160],[100, 155],[114, 151],[123, 154],
       [134, 151],[146, 154],[160, 159],[147, 167],[135, 173],[124, 175],[114, 174],[101, 170],[ 92, 161],[113, 160],[124, 161],[134, 160],[155, 160],
       [134, 160],[124, 162])
#change it to float
avglm = np.array(avglm1,dtype=float)

U=[]
for i in range(28,49):
       U.append(avglm[i])


class FaceDet():
       def __init__(self):
        self.conf =[]
        self.left =[]
        self.top =[]
        self.width =[]
        self.height = []
        self.landmarks = []

def transformation_from_points(points1, points2):
    points1 = np.array(points1,dtype=float)
    points2 = np.array(points2,dtype=float)
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
 
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
 
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
 
    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    #M=np.vstack([M, np.matrix([0., 0., 1.])])
    return M


def warp_im(M, im):
    output_im = cv2.warpAffine(im, M, (224,224), borderValue=[127, 127, 127])
    output_im = cv2.resize(output_im, (128,128))
    return output_im


def face_detector(lm_path, img):
       dets, scores, idx = detector.run(img, 1)
       facedet = FaceDet()
       for j, d in enumerate(dets):
              conf = scores[j]
              if conf >= 0.25 :
                  fleft   = d.left()
                  ftop    = d.top()
                  fwidth  = d.width()
                  fheight = d.height()
                  d = dlib.rectangle(left=fleft, top=ftop, right=fwidth+fleft, bottom=fheight+ftop)
                  shape = predictor(img, d)
                  lmark = [[shape.part(i).x,shape.part(i).y] for i in range(0,68)]
                  facedet.conf=conf
                  facedet.left=fleft
                  facedet.top=ftop
                  facedet.width=fwidth
                  facedet.height=fheight
                  facedet.landmarks=lmark
       return facedet

def create_video_folders(main_PATH, name_id):
    """Creates a directory for each label name in the dataset."""
    name_id_path = os.path.join(main_PATH, name_id)
    if not os.path.exists(name_id_path):
        os.makedirs(name_id_path)
    return name_id_path



if (not os.path.exists(save_path)):           # C:/Users/cmysl/Desktop/new_data/
    os.mkdir(save_path)


for i in range (0,len(spearker_names_list)):
    save_dir = create_video_folders(save_path, str(i))  #D:/GRID_new_data/0
    speaker_dir = os.path.join(main_path , spearker_names_list[i])                           # 'D:/GRID/s1'
    video_names_dir=os.listdir(speaker_dir)
    video_names_dir.sort()
    vid=0
    for item in video_names_dir:
        if item.endswith('.mpg'):
            video_names=os.path.join(speaker_dir,item)      #'D:/GRID/s1/bbaf2n.mpg'
            ABOUT_00001_dir = create_video_folders(save_dir, str(vid))     # D:/GRID_new_data/0/0'
            # create folders for data
            align_face_dir = create_video_folders(ABOUT_00001_dir, "align_face")   #C:/Users/cmysl/Desktop/new_data/test\\0\\0\\align_face
            mfcc35_dir = create_video_folders(ABOUT_00001_dir, "mfcc35")  # C:/Users/cmysl/Desktop/new_data/test\\0\\0\\mfcc35
            flow_dir = create_video_folders(ABOUT_00001_dir, "flow")  # C:/Users/cmysl/Desktop/new_data/test\\0\\0\\flow
            identity_dir = create_video_folders(ABOUT_00001_dir, "identity_image")   #C:/Users/cmysl/Desktop/new_data/test\\0\\0\\identity_image
            #capture video
            vidcap = cv2.VideoCapture(video_names)  
            n_flow=0
            count=0
            for count in range(54):
                success,image = vidcap.read()
                if success:
                    if count<4:            #in order to align mfcc35 from fifth frame
                        pass
                    else:
                        facedet= face_detector(lm_path,image)
                        #change it to float
                        facedet.landmarks = np.array(facedet.landmarks,dtype=float)
                        if(len(facedet.landmarks)==0):
                            break;
                        facedet.landmarks=facedet.landmarks.tolist()
                        V=[]
                        for i in range(28,49):
                            V.append(facedet.landmarks[i])
                        M=transformation_from_points(V,U)
                        image = np.array(image,dtype=float)
                        align_img=warp_im(M,image)
                        align_img=np.array(align_img,dtype='uint8')
                        cv2.imwrite(os.path.join(align_face_dir, str(count-4) + ".jpg"), align_img)
                        if(count==4):
                            cv2.imwrite(os.path.join(identity_dir, str(count-4) + ".jpg"), align_img)
                        face2 = cv2.cvtColor(align_img, cv2.COLOR_BGR2GRAY)                  # grey scale image
                        flow = 0
                        if (n_flow >= 1):
                            flow = cv2.calcOpticalFlowFarneback(prvs2,face2,None,0.5,3,10,5,5,1.1,0)#add a parameter 'None'
                            flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
                            flow = flow.astype(np.uint8)
                            flow = np.concatenate((flow, np.zeros((128, 128, 1))), 2)
                            cv2.imwrite(os.path.join(flow_dir, str(count-4) + ".jpg"), flow)               # save flow files
                        prvs2 = face2
                        n_flow += 1
                else:
                    break
            # can extract all frames
            if n_flow==50:
                #Extract wav file
                command = ['ffmpeg -i', video_names,
                             '-f wav -acodec pcm_s16le -ar 16000',
                            os.path.join(ABOUT_00001_dir, str(vid) + ".wav")]
                command = ' '.join(command)
                try:
                    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as err:
                    print(err)
                wav_file=os.path.join(ABOUT_00001_dir, str(vid) + ".wav")
                wav_to_mfcc(wav_file,mfcc35_dir)
                print("converted " + speaker_dir + " " + str(vid) + ".mp4")
                vid+=1
            else:
                #delete the current folder
                shutil.rmtree(ABOUT_00001_dir)