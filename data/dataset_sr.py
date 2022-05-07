import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util


class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        # 获取到训练集参数
        self.opt = opt
        # 获取输入通道数默认为3
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        # 获取下采样倍数
        self.sf = opt['scale'] if opt['scale'] else 4
        # 获取HR图片的大小，获取要求的大小，后面对图片进行裁剪，变成要求的大小
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        # 计算出低分辨大小
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        # 获取路径高分与低分
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L),
                                                                                           len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        # 获取高分图片
        H_path = self.paths_H[index]
        # 读取图片
        img_H = util.imread_uint(H_path, self.n_channels)
        # 将图片归一化 /255
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        # 将图片裁剪成采样倍数的整倍数，方便采样
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        # 获取低质量图片
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            # 如果有低质量图片的路径，则去读取图片
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)
        # 没有低质量的图片路径，通过下采样去获取低质量的图片，通过matlab的bicubic 没有特殊模糊
        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            # 获取低质量图片，
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            # 如果是在训练，那么将L_img 的大小随机裁剪成指定大小
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            # 通过L_img的裁剪参数获得对应的H_img
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            # 对L/H图片进行相同的随机操作，有七种随机的翻转操作，对
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        # 将nparray转化层tensor CHW格式
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
