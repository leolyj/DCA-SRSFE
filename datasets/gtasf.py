import sys, os
import os.path as osp
import numpy as np
import glob
import random

import torch.utils.data as data


__all__ = ['gtasf']


class gtasf(data.Dataset):
    def __init__(self,
                 train,
                 transform,
                 gen_func,
                 args
                 ):
        self.root = osp.join(args.data_root, 'GTA-SF')
        self.train = train
        self.transform = transform
        self.gen_func = gen_func
        self.num_points = args.num_points
        self.remove_ground = args.remove_ground

        self.samples = []
        if self.train:
            self.sample_sequence_list = ['00', '01']  # train on part of the dataset for efficiency
            # self.sample_sequence_list = ['00', '01', '02', '03', '04', '05']
        else:
            self.sample_sequence_list = []

        for s in range(len(self.sample_sequence_list)):
            sequence_name_k = os.path.splitext(self.sample_sequence_list[s])[0]  # segment-xxx
            npz_dir = os.path.join(self.root, sequence_name_k)
            npz_path = glob.glob(os.path.join(npz_dir, '*.npz'))
            self.samples += npz_path

        print('len samples', len(self.samples))

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fn = self.samples[index]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pc1'].astype(np.float32)  # 1st frame
            pc2 = data['pc2'].astype(np.float32)  # 2nd frame
            flow = data['flow'].astype(np.float32)  # scene flow label
            pos1 = data['pos1']  # ego-position of 1st frame
            pos2 = data['pos2']  # ego-position of 2nd frame
            rot1 = data['rot1']  # ego-rotation
            rot2 = data['rot2']
            is_ground_1 = data['is_ground0']  # 0 - not_ground, 1 - is_ground
            is_ground_2 = data['is_ground1']

        # remove points far away from ego-car
        not_outrange_1 = np.square(pc1[:, 0]) + np.square(pc1[:, 1]) + np.square(pc1[:, 2]) < 60 * 60
        not_outrange_2 = np.square(pc2[:, 0]) + np.square(pc2[:, 1]) + np.square(pc2[:, 2]) < 60 * 60
        not_outbound_1 = abs(pc1[:, 1]) < 50
        not_outbound_2 = abs(pc2[:, 1]) < 50
        # remove ground points
        # not_ground_1 = pc1[:, 2] > -1.65
        # not_ground_2 = pc2[:, 2] > -1.65
        not_ground_1 = (is_ground_1 == 0)
        not_ground_2 = (is_ground_2 == 0)

        # keep the selected points
        pc1_transformed, pc2_transformed = pc1[not_outrange_1 * not_ground_1 * not_outbound_1, :], pc2[not_outrange_2 * not_ground_2 * not_outbound_2, :]
        sf_transformed = flow[not_outrange_1 * not_ground_1 * not_outbound_1, :]

        n1, n2 = pc1_transformed.shape[0], pc2_transformed.shape[0]

        # choose another sample if the selected results are empty (too far away or too much ground)
        if (n1 == 0) | (n2 == 0):
            print('path {} get n=0'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        # sample to num_points
        if n1 >= self.num_points:
            sample_idx1 = np.random.choice(n1, self.num_points, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.num_points - n1, replace=True)), axis=-1)
        if n2 >= self.num_points:
            sample_idx2 = np.random.choice(n2, self.num_points, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.num_points - n2, replace=True)), axis=-1)

        pc1_transformed = pc1_transformed[sample_idx1, :]
        pc2_transformed = pc2_transformed[sample_idx2, :]
        sf_transformed = sf_transformed[sample_idx1, :]
        pc1_transformed = np.vstack((pc1_transformed[:, 1], pc1_transformed[:, 2], pc1_transformed[:, 0])).T
        pc2_transformed = np.vstack((pc2_transformed[:, 1], pc2_transformed[:, 2], pc2_transformed[:, 0])).T
        sf_transformed = np.vstack((sf_transformed[:, 1], sf_transformed[:, 2], sf_transformed[:, 0])).T

        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1, pc2, sf, generated_data = self.gen_func([pc1_transformed,
                                                      pc2_transformed,
                                                      sf_transformed])

        return pc1, pc2, sf, generated_data, self.samples[index]

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

