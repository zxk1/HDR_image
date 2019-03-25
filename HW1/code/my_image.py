import sys
import time
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

from load_radiance_and_psnr import load_radiance,PSNR_UCHAR3
from hdr_my import global_tone_mapping,local_tone_mapping,gaussian,bilateral

# ignore warning
import warnings
warnings.filterwarnings("ignore")

image_path = ("../TestImg/my_image_2.hdr") 
 
#%% global tone mapping
print("start global tone mapping...")

# load radiance(*.hdr)
HDRIMG = load_radiance(image_path)
np.multiply(HDRIMG,100,HDRIMG)
# tone mapping
start = time.time()
img_global_tone_map = global_tone_mapping(HDRIMG, WB='False')
end = time.time()
period = end - start
print("process time = %f sec"%period)

# save and show image
scipy.misc.imsave("../result/my_image_global_tone.png",img_global_tone_map)
#plt.figure()
#plt.imshow(img_global_tone_map)

'''
Please do the following parts by yourself. 
Format is not restricted. Just make sure it can test the correctness of your code.
'''
#%% local tone mapping with gaussian filter
HDRIMG = load_radiance(image_path)

start = time.time()
img_gaussian = local_tone_mapping(HDRIMG,Filter=gaussian,window_size=35,sigma_s=100,sigma_r=0.8)
end = time.time()
period = end - start
print("process time = %f sec"%period)

# save and show image
scipy.misc.imsave("../result/my_image_gaussian.png",img_gaussian)
#plt.figure()
#plt.imshow(img_gaussian)

#%% local tone mapping with bilateral filter
HDRIMG = load_radiance(image_path)

start = time.time()
img_bilateral = local_tone_mapping(HDRIMG,Filter=bilateral,window_size=35,sigma_s=100,sigma_r=0.8)
end = time.time()
period = end - start
print("process time = %f sec"%period)

# save and show image
scipy.misc.imsave("../result/my_image_bilateral.png",img_bilateral)
#plt.figure()
#plt.imshow(img_bilateral)

#%% global tone mapping with white balance
HDRIMG = load_radiance(image_path)
np.multiply(HDRIMG,100,HDRIMG)
start = time.time()
img_global_wb = global_tone_mapping(HDRIMG,WB='True')
end = time.time()
period = end - start
print("process time = %f sec"%period)

# save and show image
scipy.misc.imsave("../result/my_image_global_wb.png",img_global_wb)
#plt.figure()
#plt.imshow(img_global_wb)
