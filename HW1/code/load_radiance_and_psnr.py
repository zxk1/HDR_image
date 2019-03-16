import numpy as np

def bytestoint(byte_in):
    """convert byte(0~255) to int"""
    return ord(byte_in)

def load_radiance(filepath):
    """load radiance map from input path and return Mat<float>"""
    with open(filepath, mode='rb') as rad_file:
        while 1:
            rawline = rad_file.readline()
            if rawline[0] == '-' or rawline[0] == '+':
                splitted = rawline.split()
                size_y = int(splitted[1].decode("utf-8"))
                size_x = int(splitted[3].decode("utf-8"))
                break
        rgbe = np.zeros((size_y, size_x, 4), np.uint8, 'C')
        for row_idx in range(size_y):
            buf = rad_file.read(4)
            if len(buf) != 4:
                break
            for channel_idx in range(4):
                index_cnt = 0
                while index_cnt < size_x:
                    tmp = rad_file.read(1)
                    runlength = bytestoint(tmp)
                    if runlength <= 128:
                        for _ in range(runlength):
                            pix_in = bytestoint(rad_file.read(1))
                            rgbe[row_idx, index_cnt, channel_idx] = pix_in
                            index_cnt = index_cnt + 1
                    else:
                        pix_in = bytestoint(rad_file.read(1))
                        rgbe[row_idx, index_cnt:index_cnt+runlength-128, channel_idx] = pix_in #129 to 128
                        index_cnt = index_cnt + runlength -128
    hdrimage = np.zeros((size_y, size_x, 3), float)
    value = rgbe[:, :, 3] # get exponent
    value = value.astype(int)
    value = value - 128
    value_normalize = np.ldexp(1.0/256.0,value)
    for _ in range(3):
        hdrimage[:, :, _] = (rgbe[:, :, _] + 0.5)*value_normalize
    select = rgbe[:, :, 3] == 0    # 2D true-false table
    hdrimage[select] = [0, 0, 0]  # if select's pixel is false, then let RGB value equal to zero
    return hdrimage

def PSNR_UCHAR3(input_1, input_2, peak=255):
    [row,col,channel] = input_1.shape
    if input_1.shape != input_2.shape:
        print("Warning!! Two image have different shape!!")
        return 0
    mse = ((input_1 - input_2)**2).sum() / (row * col * channel)
    
    return 20*np.log10(peak) - 10*np.log10(mse)