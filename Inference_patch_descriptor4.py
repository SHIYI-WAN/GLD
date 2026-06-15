from PIL import Image
import os
import cv2
import torch
import argparse
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from Patch_gabor_descriptor_drop_one_block_5_5_last2Block import Gabor_descriptor

parser = argparse.ArgumentParser(description='inference code')

# --------------------------------------------------------------------------------------------------------
# ----------------------------------------- INPUT PARAM --------------------------------------------------
# --------------------------------------------------------------------------------------------------------
parser.add_argument('--resume', default='/project/project/shiyi/home_matchformer/MatchFormer/GB_DHT_deconv/checkp/b256_lyft_1222_GB_patch_descriptor_onepair_kom_lr01_wd_4_SGD_drop_one_block_last3Block_cut_5_5/checkpoint_100.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--output_location', default='/project/project/shiyi/lyft_training_data/Adapted_Data_out2_rancac_sq/evaluation_all_det/b256_lyft_1222_GB_patch_descriptor_onepair_kom_lr01_wd_4_SGD_drop_one_block_last3Block_cut_5_5_100ep_same_192128/descriptor/', type=str, metavar='PATH', help='path of the output folder')
parser.add_argument('--patch_location', default='/project/project/shiyi/lyft_training_data/Adapted_Data_out2_rancac_sq/patch_new_2d_det2_192128_all/Cluster_3/', type=str, metavar='PATH', help='path of patches')


def get_descriptors(weights_path, Patches_location , Output_location):
    global args
    args = parser.parse_args()
    transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize((0.492967568115862), (0.272086182765434)
                                    ])

    # --------------------------------------------------------------------------------------------------------
    # ----------------------------------------- LOAD THE NETWORK WEIGHTS -------------------------------------
    # --------------------------------------------------------------------------------------------------------
    # if os.path.isfile(weights_path):
    #     model = torch.load(weights_path).cuda()
    # else:
    #     print("=> no checkpoint found at '{}'".format(weights_path))
    #     return
    # --------------------------------------------------------------------------------------------------------
    model = Gabor_descriptor().cuda()
    # model = model.to('cuda')
    # model = torch.nn.DataParallel(model).cuda()

    # --------------------------------------------------------------------------------------------------------
    # ----------------------------------------- LOAD THE WEIGHTS ---------------------------------------------
    # --------------------------------------------------------------------------------------------------------
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    # set the model to evaluation mode
    model.eval()

    # read all folder names (image names) that contain patches
    # --------------------------------------------------------------------------------------------------------
    # ----------------------------------------- READ THE IMAGE FOLDERS ---------------------------------------
    # --------------------------------------------------------------------------------------------------------
    dirs = os.listdir(Patches_location)
    count=1
    for folder_name in dirs:

            # check if the output folder exist, if not create the output folder
            if not os.path.exists(Output_location + folder_name):
                os.makedirs(Output_location + folder_name)

            # list the image folders
            listPng = os.listdir(Patches_location + folder_name)

            # loop through the patches and provide the 128 descriptor that is saved in a text file
            # --------------------------------------------------------------------------------------------------------
            # ----------------------------------------- PROVIDE THE DESCRIPTOR ---------------------------------------
            # --------------------------------------------------------------------------------------------------------
            for i in range(0,len(listPng)):

                # patch path (directory/image name/ patch id .ext)
                PngName = Patches_location +'/'+ folder_name +'/'+ listPng[i][:-4] + '.png'

                # read the patch
                # im = cv2.imread(PngName,0)
                # img1_torch = transform(im)

                im = Image.open(PngName).convert("RGB")
                img1_torch =  transform(im)

                img1_torch = img1_torch.unsqueeze(0)
                img1_torch = Variable(img1_torch.float().cuda())
                y = model(img1_torch)

                # get the 128D descriptor array
                d1 = y.squeeze(0).detach().cpu().numpy()

                # write the descriptor in ta texte file such that each descriptor values are stored line by line
                np.savetxt(Output_location + folder_name +'/'+ listPng[i][:-4]+ ".txt", d1, fmt="%s")

            # print number of folders(images) / number of lines (patches) / folder name (image name)
            print(str(count) + ' ' + str(len(listPng)) + ' ' + str(folder_name))
            count= count + 1

#--------------------------------------------------------------------------------------------------------
#----------------------------------------- RUN INFERENCE ------------------------------------------------
#--------------------------------------------------------------------------------------------------------
def main():

    args = parser.parse_args()

    # --------------------------------------------------------------------------------------------------------
    # ------------------------------ SET THE INPUT/OUTPUT FOLDER + NET WEIGHT --------------------------------
    # --------------------------------------------------------------------------------------------------------
    NetWorkThLocation = args.resume
    Patches_location = args.patch_location
    Output_location = args.output_location

    if not os.path.exists(Output_location):
        os.mkdir(Output_location)

    get_descriptors(weights_path=NetWorkThLocation, Patches_location=Patches_location, Output_location=Output_location)

if __name__ == '__main__':
    main()