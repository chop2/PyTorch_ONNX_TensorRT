import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes
import tensorrt as trt
import cv2
from torchvision import datasets, transforms
import os
import copy
import math
import argparse
from tqdm import trange


def RectOverlap(Rect1, Rect2):
    (x11, y11, x12, y12) = Rect1
    (x21, y21, x22, y22) = Rect2
    StartX = min(x11, x21)
    EndX = max(x12, x22)
    StartY = min(y11, y21)
    EndY = max(y12, y22)
    CurWidth = (x12 - x11) + (x22 - x21) - (EndX - StartX)
    CurHeight = (y12 - y11) + (y22 - y21) - (EndY - StartY)

    if CurWidth <= 0 or CurHeight <= 0:
        return []
    else:
        X1 = max(x11, x21)
        Y1 = max(y11, y21)
        X2 = min(x12, x22)
        Y2 = min(y12, y22)
        Area = CurWidth * CurHeight  # area
        IntersectRect = (X1, Y1, X2, Y2)
        return IntersectRect

def safe_crop(img, src_rect_in):
    # swap the heitgh and width
    src_rect = copy.deepcopy(src_rect_in)
    src_rect[0], src_rect[1] = src_rect[1], src_rect[0]
    ##begin crop
    img_rect = [0, 0, img.shape[0], img.shape[1]]
    crop_rect = [int(src_rect[0] - src_rect[2] * 0.5), int(src_rect[1] - src_rect[3] * 0.5),
                 int(src_rect[0] + src_rect[2] * 0.5), int(src_rect[1] + src_rect[3] * 0.5)]
    union_rect = RectOverlap(img_rect, crop_rect)
    union_width = union_rect[2] - union_rect[0]
    union_height = union_rect[3] - union_rect[1]
    img_new = np.zeros((src_rect[2], src_rect[3], 3), dtype=np.uint8)
    # img_out = img_arr_src[ x_b:self.crop_h + x_b , x_b:self.crop_w + x_b , : ]
    img_new[union_rect[0] - crop_rect[0]: union_rect[0] - crop_rect[0] + union_width,
    union_rect[1] - crop_rect[1]:  union_rect[1] - crop_rect[1] + union_height, :] = img[union_rect[0]: union_rect[2],
                                                                                     union_rect[1]: union_rect[3], :]
    return img_new

def crop_resize_img(img, landmark_center, jitter, crop_p_in, resize_p_in):
    img_res = None
    jit_x, jit_y = jitter
    crop_p = copy.deepcopy(crop_p_in)
    resize_p = copy.deepcopy(resize_p_in)
    len_crop = len(crop_p)
    len_resize = len(resize_p)
    assert math.fabs(len_crop - len_resize) <= 1
    tmp_img = img.copy()
    if (len_resize >= len_crop):
        for i in range(len_crop):
            crop_rect = crop_p[i]
            resize_rect = resize_p[i]
            if (crop_rect[0] == crop_rect[1] and crop_rect[0] == -1):
                crop_rect[0] = int(tmp_img.shape[0] * 0.5) + jit_x
                crop_rect[1] = int(tmp_img.shape[1] * 0.5) + jit_y
            else:
                crop_rect[0] = landmark_center[0] + jit_x
                crop_rect[1] = landmark_center[1] + jit_y
            tmp_img = safe_crop(tmp_img, crop_rect)
            tmp_img = cv2.resize(tmp_img, (resize_rect[0], resize_rect[1]))

        resize_rect = resize_p[-1]
        tmp_img = cv2.resize(tmp_img, (resize_rect[0], resize_rect[1]))
    else:
        for i in range(len_resize):
            crop_rect = crop_p[i]
            resize_rect = resize_p[i]
            if (crop_rect[0] == crop_rect[1] and crop_rect[0] == -1):
                crop_rect[0] = int(tmp_img.shape[0] * 0.5) + jit_x
                crop_rect[1] = int(tmp_img.shape[1] * 0.5) + jit_y
            else:
                crop_rect[0] = landmark_center[0] + jit_x
                crop_rect[1] = landmark_center[1] + jit_y
            # print "crop", crop_rect
            tmp_img = safe_crop(tmp_img, crop_rect)
            tmp_img = cv2.resize(tmp_img, (resize_rect[0], resize_rect[1])).copy()

        crop_rect = crop_p[-1]
        if (crop_rect[0] == crop_rect[1] and crop_rect[0] == -1):
            crop_rect[0] = landmark_center[0] + jit_x
            crop_rect[1] = landmark_center[1] + jit_y
        # print "final crop", crop_rect
        tmp_img = safe_crop(tmp_img, crop_rect)

    return tmp_img



def pre_process(img, lmks=None):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    aligned_img = img
    if (lmks is not None):
        aligned_img, warped_lmks = aligment_image(img, lmks, (232, 232))
    height, width, _ = aligned_img.shape
    crop_p = [[-1, -1, 212, 212], [64, 70, 116, 100]]
    resize_p = [[128, 128]]
    landmark_center = [int(width * 0.5), int(height * 0.5)]
    img_res = crop_resize_img(aligned_img, landmark_center, (0, 0), crop_p, resize_p)
    #cv2.imshow("show_crop", img_res)
    #cv2.waitKey(-1)
    # Post process
    img_res = img_res[:, :, (2, 1, 0)]  # bgr -> rgb
    img_res = img_res.astype('float32', copy=False)
    img_res = (img_res * 0.00392156 - mean) / std  # norm to (-1 , 1)
    img_res = img_res.transpose((2, 0, 1))
    return img_res


class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_layers, stream):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.input_layers = input_layers
        self.stream = stream
        self.cache_file = "centernet_objdet_400_calibrationTable"

        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        stream.reset()


    def get_batch_size(self):
        return self.stream.batch_size


    def get_batch(self, bindings, names):
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        for i in self.input_layers[0]:
            assert names[0] != i

        bindings[0] = int(self.d_input)
        return bindings

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()



    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)



class ImageBatchStream():
    def __init__(self, args):
        self.fileslist = args.calib_list
        self.batch_size = args.batch_size
        self.channel = args.input_channel
        self.width = args.input_width
        self.height = args.input_height
        self.batch_cache_dir = args.batch_cache_dir

        calibration_files = [l.rstrip("\n") for l in open(self.fileslist).readlines()]

        #calibration_files = calibration_files[:100]
        print("#images: ", len(calibration_files))
        self.max_batches = (len(calibration_files) // self.batch_size) + \
                           (1 if (len(calibration_files) % self.batch_size)
                            else 0)
        #self.max_batches = len(calibration_files)
        self.files = calibration_files
        self.calibration_data = np.zeros((self.batch_size, self.channel,
            self.height, self.width), dtype=np.float32)
        self.batch = 0
        # self.preprocessor = preprocessor


    def reset(self):
        self.batch = 0


    def next_batch(self):
        if self.batch < self.max_batches:
            #print("[ImageBatchStream] Processing ", self.batch, self.max_batches)
            imgs = []
            npy_dat=None
            files_for_batch = self.files[self.batch_size * self.batch: \
                                         self.batch_size * (self.batch + 1)]
            #files_for_batch = self.files[self.batch:self.batch + 1]
            for f in files_for_batch:
                #print('read file:%s' % f)
                img = cv2.imread(f)
                img = pre_process(img)

                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #img = cv2.resize(img, (self.width, self.height))
                #channel_swap = (2, 0, 1)

                #img = img.transpose(channel_swap)  # swap channel

                # img = read_image_chw(os.path.join(self.prefix, f), self.width, self.height)
                imgs.append(img)
                #npy_dat = np.load(f)
            #imgs = list(npy_dat)
            print("[ImageBatchStream] Processing ", self.batch, self.max_batches,"len(imgs)",len(imgs))
            if(False and self.batch_cache_dir != ''):
                if not os.path.isdir(self.batch_cache_dir):
                   os.makedirs(self.batch_cache_dir)
                batch_name = os.path.join(self.batch_cache_dir,'batch%d.npy' % self.batch)
                np_img = np.array(imgs)
                #np.save(batch_name,np_img)
             
            #self.calibration_data = npy_dat   
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='calib information')
    parser.add_argument('--calib_list', dest='calib_list', type=str, help='',default='calib_testset.lst.filter')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='',default=1024)
    parser.add_argument('--input_channel', dest='input_channel', type=int, help='',default=3)
    parser.add_argument('--input_width', dest='input_width', type=int, help='',default=100)
    parser.add_argument('--input_height', dest='input_height', type=int, help='',default=116)
    parser.add_argument('--batch_cache_dir',dest='batch_cache_dir',type=str,help='',default='./batch_caches_testset/')
    args = parser.parse_args()

    bs = ImageBatchStream(args)
    
    while True:
    #for i in range(0,bs.max_batches):
        batch = bs.next_batch()
        if not batch.size:
           break

    print('done.')

