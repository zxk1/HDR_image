import sys
import time
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

from load_radiance_and_psnr import load_radiance,PSNR_UCHAR3
from hdr import global_tone_mapping,local_tone_mapping,gaussian,bilateral

# ignore warning
import warnings
warnings.filterwarnings("ignore")
 
 
#%% global tone mapping
print("start global tone mapping...")

# load radiance(*.hdr)
HDRIMG = load_radiance("../TestImg/memorial.hdr")
HDR_shape = np.shape(HDRIMG)
# tone mapping
start = time.time()
img_global_tone_map = global_tone_mapping(HDRIMG, WB='False')
end = time.time()
period = end - start
print("process time = %f sec"%period)

# save and show image
scipy.misc.imsave("../result/my_memorial_global.png",img_global_tone_map)
#plt.figure()
#plt.imshow(img_global_tone_map)

# PSNR
'''
Make sure the input images of PSNR_UCHAR3 have the same color range "uint8(0-255)" or "float(0-1)".
If the type is "uint8", set the third argument(peak) of PSNR_UCHAR3 to 255, else(float), set to 1.  
'''
ground_truth_global_tone_map = scipy.misc.imread("../ref/memorial_global.png")
your_global_tone_map = scipy.misc.imread("../result/my_memorial_global.png")
PSNR_global_tone_map = PSNR_UCHAR3(your_global_tone_map,ground_truth_global_tone_map,peak=255)
print("PSNR = %f"%PSNR_global_tone_map)


'''
Please do the following parts by yourself. 
Format is not restricted. Just make sure it can test the correctness of your code.
'''
#%% local tone mapping with gaussian filter
HDRIMG = load_radiance("../TestImg/vinesunset.hdr")

start = time.time()
img_gaussian = local_tone_mapping(HDRIMG,Filter=gaussian,window_size=35,sigma_s=100,sigma_r=0.8)
end = time.time()
period = end - start
print("process time = %f sec"%period)

# save and show image
scipy.misc.imsave("../result/my_vinesunset_gaussian.png",img_gaussian)
#plt.figure()
#plt.imshow(img_gaussian)

# PSNR
ground_truth_gaussian = scipy.misc.imread("../ref/vinesunset_gaussian.png")
your_gaussian = scipy.misc.imread("../result/my_vinesunset_gaussian.png")
PSNR_gaussian = PSNR_UCHAR3(your_gaussian,ground_truth_gaussian,peak=255)
print("PSNR = %f"%PSNR_gaussian)


#%% local tone mapping with bilateral filter
HDRIMG = load_radiance("../TestImg/vinesunset.hdr")

start = time.time()
img_bilateral = local_tone_mapping(HDRIMG,Filter=bilateral,window_size=35,sigma_s=100,sigma_r=0.8)
end = time.time()
period = end - start
print("process time = %f sec"%period)

# save and show image
scipy.misc.imsave("../result/my_vinesunset_bilateral.png",img_bilateral)
#plt.figure()
#plt.imshow(img_bilateral)

# PSNR
ground_truth_bilateral = scipy.misc.imread("../ref/vinesunset_bilateral.png")
your_bilateral = scipy.misc.imread("../result/my_vinesunset_bilateral.png")
PSNR_bilateral = PSNR_UCHAR3(your_bilateral,ground_truth_bilateral,peak=255)
print("PSNR = %f"%PSNR_bilateral)


#%% global tone mapping with white balance
HDRIMG = load_radiance("../TestImg/memorial.hdr")

start = time.time()
img_global_wb = global_tone_mapping(HDRIMG,WB='True')
end = time.time()
period = end - start
print("process time = %f sec"%period)

# save and show image
scipy.misc.imsave("../result/my_memorial_global_wb.png",img_global_wb)
#plt.figure()
#plt.imshow(img_global_wb)

# PSNR
ground_truth_global_wb = scipy.misc.imread("../ref/memorial_global_wb.png")
your_global_wb = scipy.misc.imread("../result/my_memorial_global_wb.png")
PSNR_global_wb = PSNR_UCHAR3(your_global_wb,ground_truth_global_wb,peak=255)
print("PSNR = %f"%PSNR_global_wb)


