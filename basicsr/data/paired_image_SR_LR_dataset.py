# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
import torch
from torchvision.transforms.functional import normalize, resize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_hw
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import os
import numpy as np

class PairedImageSRLRDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageSRLRDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            import os
            nums_lq = len(os.listdir(self.lq_folder))
            nums_gt = len(os.listdir(self.gt_folder))

            # nums_lq = sorted(nums_lq)
            # nums_gt = sorted(nums_gt)

            # print('lq gt ... opt')
            # print(nums_lq, nums_gt, opt)
            assert nums_gt == nums_lq

            self.nums = nums_lq
            # {:04}_L   {:04}_R


            # self.paths = paired_paths_from_folder(
            #     [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #     self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']

        gt_path_L = os.path.join(self.gt_folder, '{:04}_L.png'.format(index + 1))
        gt_path_R = os.path.join(self.gt_folder, '{:04}_R.png'.format(index + 1))


        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))


        lq_path_L = os.path.join(self.lq_folder, '{:04}_L.png'.format(index + 1))
        lq_path_R = os.path.join(self.lq_folder, '{:04}_R.png'.format(index + 1))

        # lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))



        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path_L)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # if scale != 1:
        #     c, h, w = img_lq.shape
        #     img_lq = resize(img_lq, [h*scale, w*scale])
            # print('img_lq .. ', img_lq.shape, img_gt.shape)


        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': f'{index+1:04}',
            'gt_path': f'{index+1:04}',
        }

    def __len__(self):
        return self.nums // 2


def _Mixup(im1, im2, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(int(im1.size()[0])).to(im2.device)

    im1 = v * im1 + (1-v) * im1[r_index, :]
    im2 = v * im2 + (1-v) * im2[r_index, :]
    return im1, im2

def mixup(x, y, alpha=1.0):
    '''
    Mixup data augmentation.
    Args:
        x: input data.
        y: input labels.
        alpha: float, mixup ratio.
    Returns:
        mixed inputs, pairs of targets, and lambda.
    '''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def pad_to_multiple(img, multiple):
        # 计算需要padding的额外行和列
        h, w, _ = img.shape
        h_pad = (multiple - h % multiple) % multiple
        w_pad = (multiple - w % multiple) % multiple
        # 添加零padding
        img_padded = np.pad(img, [(0, h_pad), (0, w_pad), (0, 0)], mode='constant')
        return img_padded, h_pad, w_pad
# class PairedStereoImageDataset(data.Dataset):
#     '''
#     Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
#     '''
#     def __init__(self, opt):
#         super(PairedStereoImageDataset, self).__init__()
#         self.opt = opt
#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.mean = opt['mean'] if 'mean' in opt else None
#         self.std = opt['std'] if 'std' in opt else None

#         self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
#         if 'filename_tmpl' in opt:
#             self.filename_tmpl = opt['filename_tmpl']
#         else:
#             self.filename_tmpl = '{}'

#         assert self.io_backend_opt['type'] == 'disk'
#         import os
#         self.lq_files = os.listdir(self.lq_folder)
#         self.gt_files = os.listdir(self.gt_folder)

#         self.nums = len(self.gt_files)

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(
#                 self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         gt_path_L = os.path.join(self.gt_folder, self.gt_files[index])
#         gt_path_R = os.path.join(self.gt_folder, self.gt_files[index])
#         # print(gt_path_L,gt_path_R)
#         img_bytes = self.file_client.get(gt_path_L, 'gt')
#         try:
#             img_gt_L = imfrombytes(img_bytes, float32=True)
#         except:
#             raise Exception("gt path {} not working".format(gt_path_L))

#         img_bytes = self.file_client.get(gt_path_R, 'gt')
#         try:
#             img_gt_R = imfrombytes(img_bytes, float32=True)
#         except:
#             raise Exception("gt path {} not working".format(gt_path_R))

#         lq_path_L = os.path.join(self.lq_folder, self.lq_files[index])
#         lq_path_R = os.path.join(self.lq_folder, self.lq_files[index])

#         # lq_path = self.paths[index]['lq_path']
#         # print(', lq path', lq_path)
#         img_bytes = self.file_client.get(lq_path_L, 'lq')
#         try:
#             img_lq_L = imfrombytes(img_bytes, float32=True)
#         except:
#             raise Exception("lq path {} not working".format(lq_path_L))

#         img_bytes = self.file_client.get(lq_path_R, 'lq')
#         try:
#             img_lq_R = imfrombytes(img_bytes, float32=True)
#         except:
#             raise Exception("lq path {} not working".format(lq_path_R))

#         img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
#         img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

#         scale = self.opt['scale']
#         # augmentation for training
#         if self.opt['phase'] == 'train':
#             if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
#                 gt_size_h = int(self.opt['gt_size_h'])
#                 gt_size_w = int(self.opt['gt_size_w'])
#             else:
#                 gt_size = int(self.opt['gt_size'])
#                 gt_size_h, gt_size_w = gt_size, gt_size

#             if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
#                 idx = [
#                     [0, 1, 2, 3, 4, 5],
#                     [0, 2, 1, 3, 5, 4],
#                     [1, 0, 2, 4, 3, 5],
#                     [1, 2, 0, 4, 5, 3],
#                     [2, 0, 1, 5, 3, 4],
#                     [2, 1, 0, 5, 4, 3],
#                 ][int(np.random.rand() * 6)]

#                 img_gt = img_gt[:, :, idx]
#                 img_lq = img_lq[:, :, idx]

#             # random crop
#             img_gt, img_lq = img_gt.copy(), img_lq.copy()
#             img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
#                                                 'gt_path_L_and_R')
#             # mixup
#             if 'mix_up' in self.opt and self.opt['mix_up']:
#                 img_gt, img_lq = _Mixup(img_gt, img_lq)
                
#             # flip, rotation
#             imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
#                                     self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)


#             img_gt, img_lq = imgs
#         else:
#             # 非训练阶段进行padding
#             img_lq, h_pad_lq, w_pad_lq = pad_to_multiple(img_lq, 8)
#             img_gt, img_lq = img2tensor([img_gt, img_lq],
#                                         bgr2rgb=True,
#                                         float32=True)
#         # normalize
#         if self.mean is not None or self.std is not None:
#             normalize(img_lq, self.mean, self.std, inplace=True)
#             normalize(img_gt, self.mean, self.std, inplace=True)
#         if self.opt['phase'] != 'train':
#                 return {
#                     'lq': img_lq,
#                     'gt': img_gt,
#                     'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
#                     'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
#                     'h_pad':  h_pad_lq,
#                     'w_pad':  w_pad_lq,
#                 }
#         else:
#             return {
#                  'lq': img_lq,
#                     'gt': img_gt,
#                     'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
#                     'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
#                     'h_pad':  0,
#                     'w_pad':  0,
#             }

#     def __len__(self):
#         return self.nums

class PairedStereoImageDataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''
    def __init__(self, opt):
        super(PairedStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.ycc = opt['ycc'] if 'ycc' in opt else None
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        import os
        self.lq_files = os.listdir(self.lq_folder)
        self.gt_files = os.listdir(self.gt_folder)

        self.nums = len(self.gt_files)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index])
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index])

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        lq_path_L = os.path.join(self.lq_folder, self.lq_files[index])
        lq_path_R = os.path.join(self.lq_folder, self.lq_files[index])

        # lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]
                # Convert img_gt and img_lq from RGB to YCbCr format
            # random crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()
            img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
                                                'gt_path_L_and_R')
            # mixup
            if 'mix_up' in self.opt and self.opt['mix_up']:
                img_gt, img_lq = _Mixup(img_gt, img_lq)
                
            # flip, rotation
            imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)


            img_gt, img_lq = imgs
    
        else:
            # 非训练阶段进行padding
            img_lq, h_pad_lq, w_pad_lq = pad_to_multiple(img_lq, 8)
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        if self.opt['phase'] != 'train':
            
            return {
                'lq': img_lq,
                'gt': img_gt,
                'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
                'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
                'h_pad':  h_pad_lq,
                'w_pad':  w_pad_lq,
            }
        else:
            return {
                 'lq': img_lq,
                    'gt': img_gt,
                    'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
                    'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
                
            }

    def __len__(self):
        return self.nums