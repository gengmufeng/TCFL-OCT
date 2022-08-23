from __future__ import print_function
import argparse
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage import io
import torch
from model import *
from DataLoader_test import get_Test_Set

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default= 72, help="the name of the trained model")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--threads", type=int, default=0, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

generator = Generator()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
torch.cuda.set_device(1)

if cuda:
    generator = generator.cuda()

print('===> Loading datasets')

val_dataloader = get_Test_Set()
validation_data_loader = DataLoader(dataset=val_dataloader, num_workers=opt.threads, batch_size=opt.batch_size,
                                    shuffle=False)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def test():
    generator.load_state_dict(torch.load(r"../result/saved_models/generator_%d.pth" % opt.epoch, map_location='cuda:1'))

    index = 0
    for i, batch in enumerate(validation_data_loader):
        index = index + 1
        noisy = Variable(batch["A"])
        if cuda:
            noisy = noisy.cuda()
        with torch.no_grad():
            noise = generator(noisy)
            imgout_test = noisy - noise
            imgout_test = (imgout_test[0][0]).detach().cpu().numpy()
            imgout_test[imgout_test>1.0]=1.0
            imgout_test[imgout_test<0.0]=0.0

        imgout_test = imgout_test * 255.0
        root_result_test = r"../result/test/"
        filename_result_test = str(index) + '_test.tif'
        filename_abs_root_test = os.path.join(root_result_test, filename_result_test)
        io.imsave(filename_abs_root_test, imgout_test)
        
test()