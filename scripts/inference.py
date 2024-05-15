import torch
from omegaconf import OmegaConf
import sys
import os
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_ldm_ref_inpaint import DDIMSampler
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from omegaconf import OmegaConf
import configargparse
import pdb
import cv2
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from tcn import TemporalConvNet
from networks import NetworkBase
import torch.nn as nn

class BasicBlock_AU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(1,1), stride=(1,1),maxpool=(2,2), padding=0):
        super(BasicBlock_AU, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.bn1(out)
        out = self.relu(out)
        return out
class RNNModel(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, num_layers=2,batch_first=True,bidirectional=True):
        super(RNNModel, self).__init__()
        #self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = num_layers
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers,
                           batch_first=True,
                           bidirectional=True)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, inputs):

        output,_ = self.rnn(inputs)
        return output
class Audio2AU(nn.Module):
    def __init__(self, hidden_size=128):
        super(Audio2AU, self).__init__()
        
        # the input map is 1 x 12 x 28       
        self.block1 = BasicBlock_AU(1, 8, (1,4), stride=(1,2),maxpool=(1,1)) # 3 x 12 x 13
        self.block2 = BasicBlock_AU(8, 32, kernel_size=(1,3), stride=(1,2),maxpool=(1,1)) # 8 x 12 x 6
        self.block3 = BasicBlock_AU(32, 64, kernel_size=(1,3), stride=(1,2),maxpool=(1,2)) # 16 x 12 x 1
        self.block4 = BasicBlock_AU(64, 128, kernel_size=(3,1), stride=(2,1),maxpool=(1,1)) # 32 x 5 x 1
        self.block5 = BasicBlock_AU(128, 256, kernel_size=(3,1), stride=(2,1),maxpool=(2,1)) # 32 x 1 x 1 
        self.rnn = RNNModel(256, hidden_size)
        self.fc1 = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5)) # 128
        self.fc2 = nn.Sequential(nn.Linear(128,64), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5)) # 128

              
    def forward(self, audio_inputs):
        batchsize=audio_inputs.shape[0]
        seq_len=audio_inputs.shape[1]
        audio_inputs = audio_inputs.contiguous().view(audio_inputs.shape[0]*audio_inputs.shape[1], audio_inputs.shape[2], audio_inputs.shape[3],audio_inputs.shape[4])
        out = self.block1(audio_inputs)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        out = out.contiguous().view(out.shape[0], -1)
        out = out.contiguous().view(batchsize,seq_len, -1)
        out = self.rnn(out)
        rnn_out = out.contiguous().view(batchsize*seq_len, -1)
        out = self.fc1(rnn_out)
        out = self.fc2(out)

        return rnn_out,out
        
        
from networks import NetworkBase


class Generator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64,  repeat_num=6):
        super(Generator, self).__init__()
        self._name = 'generator_wgan'

        layers = []
        layers.append(nn.Conv2d(3+5, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        self.attetion_reg = nn.Sequential(*layers)

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        #print("c",c.shape)64
        #cc = c.unsqueeze(0).expand(x.shape[0], -1)
        #print(x.shape,c.shape)
        cc = c.unsqueeze(2).unsqueeze(3)
        #print("cc",cc.shape)4 64 1 1
        cc = cc.expand(cc.size(0), cc.size(1), x.size(2), x.size(3))
        #print("cc",cc.shape)4 64 450 450
        #x = x.contiguous().view(x.shape[0],x.shape[3],x.shape[1], x.shape[2])
        x = torch.cat([x, cc], dim=1)
        
        features = self.main(x)
        #print("features:",features.shape)# 4 64 450 450
        img=self.img_reg(features)
        #img = img.contiguous().view(img.shape[0],img.shape[2],img.shape[3], img.shape[1])
        att=self.attetion_reg(features)
        #att = att.contiguous().view(att.shape[0],att.shape[2],att.shape[3], att.shape[1])
        #print("img:",img.shape)4 3 450 450
        #print("att:",att.shape)
        return img, att

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    # elif classname.find('GRU') != -1 or classname.find('LSTM') != -1:
    #     m.weight.data.normal_(0.0, 0.02)
    #     m.bias.data.fill_(0.01)
    else:
        print(classname)
def _do_if_necessary_saturate_mask(m, saturate=False):
    return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m
def read_cv2_img(path):
    '''
    Read color images
    :param path: Path to image
    :return: Only returns color images
    '''
    img = cv2.imread(path, -1)
    #if img.shape[0] <450:
    #    img=align_face(img)
        

    if img is not None:
        if len(img.shape) != 3:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img 

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1,)
    parser.add_argument('--numworkers', type=int, default=12,)
    parser.add_argument('--save_dir', type=str, default='/fs1/home/tjuvis_2022/lxx/qikan+diff/test/', )

    return parser

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_model():
    config = OmegaConf.load("./configs/latent-diffusion/talking-inference.yaml")
    model = load_model_from_config(config, "/fs1/home/tjuvis_2022/lxx/qikan+diff/logs/2023-09-02T23-12-27_talking/checkpoints/model.ckpt")
    return model


parser = config_parser()
args = parser.parse_args()

model = get_model()
sampler = DDIMSampler(model)


ddim_steps = 200
ddim_eta = 0.0
use_ddim = True
log = dict()
samples = []
samples_inpainting= []
xrec_img = []


# init and save configs
config_file= 'configs/latent-diffusion/talking-inference.yaml'
config = OmegaConf.load(config_file)
data = instantiate_from_config(config.data)
dataset_configs = config.data['params']['validation']
datasets = dict([('validation', instantiate_from_config(dataset_configs))])


def load_ckpt(model, ckpt_path, prefix=None):
    old_state_dict = torch.load(ckpt_path)
    cur_state_dict = model.state_dict()
    for param in cur_state_dict:
        if prefix is not None:
            old_param = param.replace(prefix, '')
        else:
            old_param = param
        if old_param in old_state_dict and cur_state_dict[param].size()==old_state_dict[old_param].size():
            #print("loading param: ", param)
            model.state_dict()[param].data.copy_(old_state_dict[old_param].data)
        else:
            print("warning cannot load param: ", param)




print("#### Data #####")
for k in datasets:
    print(f"{k}, {datasets[k].__class__.__name__}, {len(datasets[k])}")

val_dataloader = DataLoader(datasets["validation"], batch_size=args.batchsize, num_workers=args.numworkers, shuffle=False)

with torch.no_grad():
    for i, batch in enumerate(val_dataloader):
        samples = []
        samples_inpainting = []
        xrec_img = []
        z, c_audio, c_lip, c_ldm, c_mask, x, xrec, xc_audio, xc_lip,gt_aus1,img1= model.get_input(batch, 'image',
                                                                      return_first_stage_outputs=True,
                                                                      force_c_encode=True,
                                                                      return_original_cond=True,
                                                                      bs=55)

        #print(c_mask.shape,xc_audio.shape)
        audio2AU10_model=Audio2AU(hidden_size=128).cuda()
        audio2AU14_model=Audio2AU(hidden_size=128).cuda()
        audio2AU20_model=Audio2AU(hidden_size=128).cuda()
        audio2AU25_model=Audio2AU(hidden_size=128).cuda()
        audio2AU26_model=Audio2AU(hidden_size=128).cuda()
        '''
        load_ckpt(audio2AU10_model, '/fs1/home/tjuvis_2022/lxx/qikan/save/real/audio2AU10_model1.pt')
        load_ckpt(audio2AU14_model, '/fs1/home/tjuvis_2022/lxx/qikan/save/real/audio2AU14_model1.pt')
        load_ckpt(audio2AU20_model, '/fs1/home/tjuvis_2022/lxx/qikan/save/real/audio2AU20_model1.pt')
        load_ckpt(audio2AU25_model, '/fs1/home/tjuvis_2022/lxx/qikan/save/real/audio2AU25_model1.pt')
        load_ckpt(audio2AU26_model, '/fs1/home/tjuvis_2022/lxx/qikan/save/real/audio2AU26_model1.pt')
        '''
        load_ckpt(audio2AU10_model, '/fs1/home/tjuvis_2022/lxx/qikan+diff/pt/grid/audio2AU10_model175100.pt')
        load_ckpt(audio2AU14_model, '/fs1/home/tjuvis_2022/lxx/qikan+diff/pt/grid/audio2AU14_model175100.pt')
        load_ckpt(audio2AU20_model, '/fs1/home/tjuvis_2022/lxx/qikan+diff/pt/grid/audio2AU20_model175100.pt')
        load_ckpt(audio2AU25_model, '/fs1/home/tjuvis_2022/lxx/qikan+diff/pt/grid/audio2AU25_model175100.pt')
        load_ckpt(audio2AU26_model, '/fs1/home/tjuvis_2022/lxx/qikan+diff/pt/grid/audio2AU26_model175100.pt')        

        ckpt = torch.load('/fs1/home/tjuvis_2022/lxx/DFRF-main/NeRF-att/net_epoch_30_id_G_new.tar')
        mask_model = Generator().cuda()
        mask_model.load_state_dict(ckpt['unet_state_dict2'])
        transform_list = [#transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5]),
        ]
        _transform = transforms.Compose(transform_list)
        face_real_=cv2.resize(read_cv2_img(img1),[112,112])
        face_real_ = _transform(Image.fromarray(face_real_))
        input_images1=face_real_.cuda()
        gt_aus1 = torch.tensor([item for item in gt_aus1])#torch.as_tensor(np.array(gt_aus1.astype(None))).cuda()
        gt_aus1= gt_aus1.type(torch.cuda.FloatTensor)
        fake_imgs_, fake_img_mask_ =mask_model(Variable(input_images1.unsqueeze(0)),Variable(gt_aus1.unsqueeze(0)))
        fake_img_mask_ = _do_if_necessary_saturate_mask(fake_img_mask_)
        image_numpy_ = fake_img_mask_.cpu().detach().numpy()
        image_numpy_t_ = np.transpose(image_numpy_, (0,2, 3, 1))
        image_numpy_t_ = torch.tensor(image_numpy_t_).cuda()
        mask=image_numpy_t_[0]
        mask=mask
        imag_real_mask=[]
        imag_real_mask1=[]
        c_lipID1=xc_lip.cuda()
        for t2 in range(0,c_lipID1.shape[0]):
            imag_real_mask.append(255.0*(1 - mask.permute(2,0,1)) * c_lipID1[t2])
            imag_real_mask1.append(255.0*(mask.permute(2,0,1)) * c_lipID1[t2])
        imag_real_mask=torch.stack(imag_real_mask)
        imag_real_mask=imag_real_mask               
        imag_real_mask1=torch.stack(imag_real_mask1)
        imag_real_mask1=imag_real_mask1
        xc_audio=xc_audio.cuda()
        rnn_feature10,AU10_feature=audio2AU10_model(xc_audio.permute(0,4,1,2,3))  
        rnn_feature14,AU14_feature=audio2AU14_model(xc_audio.permute(0,4,1,2,3))
        rnn_feature20,AU20_feature=audio2AU20_model(xc_audio.permute(0,4,1,2,3))
        rnn_feature25,AU25_feature=audio2AU25_model(xc_audio.permute(0,4,1,2,3))
        rnn_feature26,AU26_feature=audio2AU26_model(xc_audio.permute(0,4,1,2,3))
        AU_feature=torch.cat([AU10_feature, AU14_feature, AU20_feature, AU25_feature, AU26_feature],dim=1)
        c_imag_real_mask = model.encode_first_stage(imag_real_mask.cpu())
        c_imag_real_mask1 = model.encode_first_stage(imag_real_mask1.cpu())
        
        shape = (z.shape[1], z.shape[2], z.shape[3])
        N = min(x.shape[0], 55)
        c = {'audio': c_audio, 'lip': c_imag_real_mask,'lip1':c_imag_real_mask1, 'ldm': c_ldm, 'mask_image': c_mask,'AU_feature':AU_feature}

        b, h, w = z.shape[0], z.shape[2], z.shape[3]
        landmarks = batch["landmarks_all"][0]
        landmarks = landmarks / 4
        mask = batch["inference_mask"][0].to(model.device)
        mask = mask[:, None, ...]
        #print("hhhhhhhhhh")
        with model.ema_scope():
            samples_ddim, _ = sampler.sample(ddim_steps, N, shape, c, x0=z[:N], verbose=False, eta=ddim_eta, mask=mask)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                     min=0.0, max=1.0)
        samples_inpainting.append(x_samples_ddim)
        #print("hkkkkkhhhh")

        #save images
        samples_inpainting = torch.stack(samples_inpainting, 0)
        samples_inpainting = rearrange(samples_inpainting, 'n b c h w -> (n b) c h w')
        save_path = os.path.join(args.save_dir, 'show_t')
        #print(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for j in range(samples_inpainting.shape[0]):
            samples_inpainting_img = 255. * rearrange(samples_inpainting[j], 'c h w -> h w c').cpu().numpy()
            img = Image.fromarray(samples_inpainting_img.astype(np.uint8))
            print(save_path)
            img.save(os.path.join(save_path, '{:04d}_{:04d}.jpg'.format(i, j)))


