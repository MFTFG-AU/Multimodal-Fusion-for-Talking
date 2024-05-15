import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import pdb
import cv2
from PIL import Image
import csv
import torch
from imageio import imread
import numpy as np
def load_image(img_path):
    img = imread(img_path)
    if len(img.shape)==2:
        img = np.stack((img,)*3, axis=0)
    if len(img.shape)!=3:
        print("***** not rgb image ******", img_path, img.shape)
    return img.astype(np.float32)
class TALKBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = 600#40 #len(self.image_paths)
        image_list_path = os.path.join(data_root, 'data.txt')
        with open(image_list_path, "r") as f:
            self.image_num = f.read().splitlines()

        '''
        self.labels = {
            "frame_id": [int(l.split('_')[0]) for l in self.image_paths],
            "image_path_": [os.path.join(self.data_root, 'images', l+'.jpg') for l in self.image_paths],
            "audio_smooth_path_": [os.path.join(self.data_root, 'audio_smooth', l + '.npy') for l in self.image_paths],
            "landmark_path_": [os.path.join(self.data_root, 'landmarks', l+'.lms') for l in self.image_paths],
            "reference_path": [l.split('_')[0] + '_' + str(random.choice(list(set(range(1, int(self.image_num[int(l.split('_')[0])-1].split()[1])))-set(range(int(l.split('_')[1])-60, int(l.split('_')[1])+60)))))
                               for l in self.image_paths],
        }
        '''
        
        
        
        self.img_seq = []
        self.gt_seq = []
        self.au_seq = []
        self.audio_seq = []
        self.label_seq = []
        self.lip_coord_seq = []
        train_file='/fs1/home/tjuvis_2022/lxx/qikan+diff/data_list/grid_test.txt'#grid_train_new.txt'
        with open(train_file) as f:
            lines = f.readlines()
        for line in lines:
            elems = line.rstrip('\n').split(' ')
            self.img_seq.append(elems[0])
            self.au_seq.append(elems[0])
            self.gt_seq.append(elems[0])
            self.audio_seq.append(elems[1])

        
        
        
        
        
        
        
        
        i_p='/fs1/home/tjuvis_2022/lxx/DFRF-main/data_util/dataset/May/'
        self.labels = {
            "frame_id": [0 for l in range(0,7999)],
            "image_path_": [os.path.join(i_p, 'audio-face', '{:03d}.jpg'.format(l)) for l in range(0,7999)],
            #"audio_smooth_path_": '/fs1/home/tjuvis_2022/lxx/DFRF-main/data_util/dataset/Obama_/aud.npy',
            "landmark_path_": [os.path.join(i_p, 'crop', str(l)+'.lms') for l in range(0,7999)],
            #"reference_path": [l.split('_')[0] + '_' + str(random.choice(list(set(range(1, int(self.image_num[int(l.split('_')[0])-1].split()[1])))-set(range(int(l.split('_')[1])-60, int(l.split('_')[1])+60)))))
            #                   for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
    
        audios = np.load(os.path.join(self.audio_seq[idx],'mfcc.npy')).astype(np.float32) # np array: (seq_len, h, w, 1)
        audios = audios[:,:,3:28,:]
        #example["audio_smooth"] = audios
        input_reference_img,input_image_mask, input_inference_mask, input_landmarks,input_landmarks_all,input_scaler,input_images,input_images1, gt_images, input_audios, input_aus,input_aus1,lips =[],[],[],[],[],[], [], [], [], [], [],[],[]
        
        for i in range(0,50):
            #input_images.append([load_image(os.path.join(self.img_seq[idx], str(0)+".jpg"))])
            
            image = Image.open(os.path.join(self.gt_seq[idx], str(i)+".jpg"))
            if not image.mode == "RGB":
                image = image.convert("RGB")
            img = np.array(image).astype(np.uint8)
            image = Image.fromarray(img)
            h, w = image.size
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)
                image2 = image.resize((64, 64), resample=PIL.Image.BICUBIC)
            image = np.array(image).astype(np.uint8)
            input_images.append((image / 127.5 - 1.0).astype(np.float32))            
            
            landmarks = np.loadtxt(os.path.join(self.gt_seq[idx], str(i)+".lms"), dtype=np.float32)
            landmarks_img = landmarks[13:48]
            landmarks_img2 = landmarks[0:4]
            landmarks_img = np.concatenate((landmarks_img2, landmarks_img))
            scaler = h / self.size
            input_landmarks.append((landmarks_img / scaler))        
            input_landmarks_all.append((landmarks / scaler))
            input_scaler.append(scaler)
            
            inference_mask = np.ones((h, w))
            points = landmarks[2:15]
            points = np.concatenate((points, landmarks[33:34])).astype('int32')
            inference_mask = cv2.fillPoly(inference_mask, pts=[points], color=(0, 0, 0))
            inference_mask = (inference_mask > 0).astype(int)
            inference_mask = Image.fromarray(inference_mask.astype(np.uint8))
            inference_mask = inference_mask.resize((28, 28), resample=self.interpolation)
            inference_mask = np.array(inference_mask)
            input_inference_mask.append(inference_mask)
            
            mask = np.ones((self.size, self.size))
            mask[(landmarks[33][1] / scaler).astype(int):, :] = 0.
            mask = mask[..., None]
            image_mask = (image * mask).astype(np.uint8)
            input_image_mask.append((image_mask / 127.5 - 1.0).astype(np.float32))
        
            image_r = Image.open(os.path.join(self.img_seq[idx], str(0)+".jpg"))
            if not image_r.mode == "RGB":
                image_r = image_r.convert("RGB")
            img_r = np.array(image_r).astype(np.uint8)
            image_r = Image.fromarray(img_r)
            image_r = image_r.resize((self.size, self.size), resample=self.interpolation)
            image_r = np.array(image_r).astype(np.uint8)
            input_reference_img.append((image_r / 127.5 - 1.0).astype(np.float32))    
            
            gt_images.append([load_image(os.path.join(self.gt_seq[idx], str(i)+".jpg"))])
            input_audios.append([audios[i]])
            with open(os.path.join(self.au_seq[idx],str(i)+".csv")) as file:
                reader = csv.reader(file)
                for  row,index in enumerate(reader):
                    if row==1:
                        input_aus.append([index[26],index[28],index[31],index[33],index[34]])
                        input_aus1.append([float(index[9]),float(index[11]),float(index[14]),float(index[16]),float(index[17])])
        
        
        with open(os.path.join(self.au_seq[idx],str(0)+".csv")) as file:
            reader = csv.reader(file)
            for  row,index in enumerate(reader):
                if row==1:
                    input_aus_re=[float(index[9]),float(index[11]),float(index[14]),float(index[16]),float(index[17])]
        
        
        input_images=np.array(input_images)
        input_landmarks=np.array(input_landmarks)
        input_image_mask=np.array(input_image_mask)
        input_audios=np.array(input_audios)
        input_reference_img=np.array(input_reference_img)
        input_landmarks_all=np.array(input_landmarks_all)
        input_inference_mask=np.array(input_inference_mask)
        #input_aus=np.array(input_aus)
        #input_aus1=np.array(input_aus1)
        
        '''
        if self.use_lip:
            lips = self.lip_coord_seq[idx]  # [[x1, y1, x2, y2],...[x1, y1, x2, y2]]

        sample = {'img': input_images, 'audio': input_audios, 'au': input_aus ,'gt': gt_images,'au1': input_aus1}

        if self.transform:
            sample = self.transform(sample)
        smp={'img1': os.path.join(self.img_seq[idx], str(0)+".jpg")}
        sample.update(smp)
        return sample
        '''
        sample = {"image": input_images,"landmarks":input_landmarks, "landmarks_all":input_landmarks_all, "scaler":input_scaler, "inference_mask":input_inference_mask, "image_mask":input_image_mask, "reference_img":input_reference_img, "audio_smooth": input_audios, "au": input_aus_re, "au1": input_aus1}
        smp={'img1': os.path.join(self.img_seq[idx], str(0)+".jpg")}
        sample.update(smp)
        return sample
        
        
        
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = Image.open(example["image_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        img = np.array(image).astype(np.uint8)
        image = Image.fromarray(img)
        h, w = image.size
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            image2 = image.resize((64, 64), resample=PIL.Image.BICUBIC)

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        landmarks = np.loadtxt(example["landmark_path_"], dtype=np.float32)
        landmarks_img = landmarks[13:48]
        landmarks_img2 = landmarks[0:4]
        landmarks_img = np.concatenate((landmarks_img2, landmarks_img))
        scaler = h / self.size
        example["landmarks"] = (landmarks_img / scaler)
        example["landmarks_all"] = (landmarks / scaler)
        example["scaler"] = scaler

        #inference mask
        inference_mask = np.ones((h, w))
        points = landmarks[2:15]
        points = np.concatenate((points, landmarks[33:34])).astype('int32')
        inference_mask = cv2.fillPoly(inference_mask, pts=[points], color=(0, 0, 0))
        inference_mask = (inference_mask > 0).astype(int)
        inference_mask = Image.fromarray(inference_mask.astype(np.uint8))
        inference_mask = inference_mask.resize((64, 64), resample=self.interpolation)
        inference_mask = np.array(inference_mask)
        example["inference_mask"] = inference_mask

        #mask
        mask = np.ones((self.size, self.size))
        # zeros will be filled in
        mask[(landmarks[33][1] / scaler).astype(int):, :] = 0.
        mask = mask[..., None]
        image_mask = (image * mask).astype(np.uint8)
        example["image_mask"] = (image_mask / 127.5 - 1.0).astype(np.float32)
        #print(example["audio_smooth_path_"])

        #example["audio_smooth"] = np.load(example["audio_smooth_path_"]) .astype(np.float32)
        #audd = np.load('/fs1/home/tjuvis_2022/lxx/DFRF-main/data_util/dataset/May/aud.npy') .astype(np.float32)
        #example["audio_smooth"] = audd[i]
        #example["audio_smooth"] = np.load('/fs1/home/tjuvis_2022/lxx/DFRF-main/data_util/dataset/Obama_/aud.npy') .astype(np.float32)

        #reference_path = example["reference_path"].split('_')[0]
        image_r = Image.open(os.path.join('/fs1/home/tjuvis_2022/lxx/DFRF-main/data_util/dataset/May/audio-face/000.jpg'))
        if not image_r.mode == "RGB":
            image_r = image_r.convert("RGB")

        img_r = np.array(image_r).astype(np.uint8)
        image_r = Image.fromarray(img_r)
        image_r = image_r.resize((self.size, self.size), resample=self.interpolation)
        image_r = np.array(image_r).astype(np.uint8)
        example["reference_img"] = (image_r / 127.5 - 1.0).astype(np.float32)

        return example




        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = Image.open(example["image_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        image = Image.fromarray(img)
        h, w = image.size
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        landmarks = np.loadtxt(example["landmark_path_"], dtype=np.float32)
        landmarks_img = landmarks[13:48]
        landmarks_img2 = landmarks[0:4]
        landmarks_img = np.concatenate((landmarks_img2, landmarks_img))
        scaler = h / self.size
        example["landmarks"] = (landmarks_img / scaler)

        #mask
        mask = np.ones((self.size, self.size))
        mask[(landmarks[30][1] / scaler).astype(int):, :] = 0.
        mask = mask[..., None]
        image_mask = (image * mask).astype(np.uint8)
        example["image_mask"] = (image_mask / 127.5 - 1.0).astype(np.float32)

        example["audio_smooth"] = np.load(example["audio_smooth_path_"]).astype(np.float32)

        #add for reference
        image_r = Image.open(os.path.join(self.data_root, 'images', example["reference_path"] +'.jpg'))
        if not image_r.mode == "RGB":
            image_r = image_r.convert("RGB")

        img_r = np.array(image_r).astype(np.uint8)
        image_r = Image.fromarray(img_r)
        image_r = image_r.resize((self.size, self.size), resample=self.interpolation)
        image_r = np.array(image_r).astype(np.uint8)
        example["reference_img"] = (image_r / 127.5 - 1.0).astype(np.float32)

        return example


class TalkTrain(TALKBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="./data/data_train.txt", data_root="./data/HDTF", **kwargs)

class TalkValidation(TALKBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="./data/data_test.txt", data_root="./data/HDTF", flip_p=flip_p, **kwargs)
