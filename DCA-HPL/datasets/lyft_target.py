import sys, os
import os.path as osp
import numpy as np
import glob
import random

import torch.utils.data as data

__all__ = ['lyft_target']


def augmentation(pc):
    # rotation
    # angle = np.random.uniform(-0.01,
    #                          0.01)  # 0.005  0.01
    angle = np.random.uniform(-0.02, 0.02)  # (0.0349 ~ 2 deg)
    # angle = 0

    cosval = np.cos(angle)  # pi
    sinval = np.sin(angle)
    rot_matrix = np.array([[cosval, 0, sinval],
                           [0, 1, 0],
                           [-sinval, 0, cosval]], dtype=np.float32)

    # shift
    shifts = np.random.uniform(-0.3,
                               0.3,
                               (1, 3)).astype(np.float32)  # 0.1  0.5
    # pc = pc.dot(rot_matrix.T) + shifts
    pc = pc.dot(rot_matrix.T)
    # pc = pc + shifts
    return pc


class lyft_target(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 train,
                 transform,
                 gen_func,
                 args
                 ):
        self.root = args.data_root_target
        self.train = train
        self.transform = transform
        self.gen_func = gen_func
        self.num_points = args.num_points
        self.remove_ground = args.remove_ground

        self.samples = []

        if self.train:
            split = 'trainval/train'
        else:
            split = 'trainval/val'

        self.scenes_path = os.path.join(self.root, split)
        self.scenes_list = glob.glob(os.path.join(self.scenes_path, 'host-*'))
        print('num scenes:', len(self.scenes_list))
        for s in range(len(self.scenes_list)):
            npz_paths = glob.glob(os.path.join(self.scenes_list[s], '*.npz'))
            self.samples += npz_paths

        print('len samples', len(self.samples))
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fn = self.samples[index]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pc1'].astype(np.float32)
            pc2 = data['pc2'].astype(np.float32)
            flow = data['flow'].astype(np.float32)

        not_outrange_1 = np.square(pc1[:, 0]) + np.square(pc1[:, 1]) + np.square(pc1[:, 2]) < 60 * 60
        not_outrange_2 = np.square(pc2[:, 0]) + np.square(pc2[:, 1]) + np.square(pc2[:, 2]) < 60 * 60
        not_outbound_1 = abs(pc1[:, 0]) < 50
        not_outbound_2 = abs(pc2[:, 0]) < 50
        #
        # # change pc1 pc2 order
        # select_idx_1 = pc1_x90_idx * not_ground_1 * not_outrange_1 * not_outbound_1
        # select_idx_2 = pc2_x90_idx * not_ground_2 * not_outrange_2 * not_outbound_2
        select_idx_1 = not_outrange_1 * not_outbound_1
        select_idx_2 = not_outrange_2 * not_outbound_2
        # pc1_transformed, pc2_transformed = pc2[select_idx_2, :], pc1[select_idx_1, :]
        # sf_transformed = flow[select_idx_2, :]
        pc1_transformed, pc2_transformed = pc2[select_idx_2, :], pc1[select_idx_1, :]
        sf_transformed = flow[select_idx_2, :]

        n1, n2 = pc1_transformed.shape[0], pc2_transformed.shape[0]

        if n1 >= self.num_points:
            sample_idx1 = np.random.choice(n1, self.num_points, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.num_points - n1, replace=True)),
                                         axis=-1)
        if n2 >= self.num_points:
            sample_idx2 = np.random.choice(n2, self.num_points, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.num_points - n2, replace=True)),
                                         axis=-1)

        pc1_transformed = pc1_transformed[sample_idx1, :]
        pc2_transformed = pc2_transformed[sample_idx2, :]
        sf_transformed = sf_transformed[sample_idx1, :]
        pc1_transformed = np.vstack((- pc1_transformed[:, 1], pc1_transformed[:, 2], - pc1_transformed[:, 0])).T
        pc2_transformed = np.vstack((- pc2_transformed[:, 1], pc2_transformed[:, 2], - pc2_transformed[:, 0])).T
        sf_transformed = np.vstack((- sf_transformed[:, 1], sf_transformed[:, 2], - sf_transformed[:, 0])).T

        pc1_transformed_2 = augmentation(pc1_transformed)  # using augmentation for pc1 only

        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_, pc2_, sf_, generated_data = self.gen_func([pc1_transformed,
                                                         pc2_transformed,
                                                         sf_transformed])

        pc1_2, pc2_2, sf_2, generated_data_target_2 = self.gen_func([pc1_transformed_2,
                                                                     pc2_transformed,
                                                                     sf_transformed])
        # pc1_target, pc2_target, generated_data_target, pc1_target_2, generated_data_target_2
        return pc1_, pc2_, generated_data, pc1_2, pc2_2, generated_data_target_2, self.samples[index], sf_

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str


if __name__ == '__main__':
    pass

