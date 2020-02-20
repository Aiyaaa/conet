"""
This dataset for Bengali.AI Handwritten Grapheme Classification
https://www.kaggle.com/c/bengaliai-cv19/discussion/123757
"""

from os import path
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from skimage.color import gray2rgb
import cv2
from os import path
import copy
import pandas as pd

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import pickle


class Struct(object):
    def __init__(self, is_copy=False, **kwargs):
        self.add(is_copy, **kwargs)

    def add(self, is_copy=False, **kwargs):
        # self.__dict__.update(kwargs)

        if is_copy == False:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, copy.deepcopy(value))
                    # setattr(self, key, value.copy())
                except Exception:
                    setattr(self, key, value)

    def __str__(self):
        text = ''
        for k, v in self.__dict__.items():
            text += '\t%s : %s\n' % (k, str(v))
        return text


def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort=pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    # df = df.reset_index()
    df = df.drop('sort', axis=1)
    return df


DATA_DIR = '/data1/hangli/bengali/data'
IMAGE_HEIGHT, IMAGE_WIDTH = 137, 236

TASK = {
    'grapheme_root': Struct(
        num_class=168,
    ),
    'vowel_diacritic': Struct(
        num_class=11,
    ),
    'consonant_diacritic': Struct(
        num_class=7,
    ),
    'grapheme': Struct(
        num_class=1295,
        class_map=dict(pd.read_csv(DATA_DIR + '/grapheme_1295.csv')[['grapheme', 'label']].values),
        # freqency  = None,
    ),
}
NUM_TASK = len(TASK)
NUM_CLASS = [TASK[k].num_class for k in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme']]

TRAIN_NUM = 200840
TRAIN_PARQUET = None


class BenliDataset(Dataset):
    def __init__(self, split, mode, csv, parquet, augment=None):
        df = pd.read_csv(DATA_DIR + '/%s' % csv)
        if parquet is not None:
            uid = []
            image = []
            for f in parquet:
                d = pd.read_parquet(DATA_DIR + '/%s' % f, engine='pyarrow')
                uid.append(d['image_id'].values)
                image.append(d.drop('image_id', axis=1).values.astype(np.uint8))
            uid = np.concatenate(uid)
            image = np.concatenate(image)

        if split is not None:
            s = np.load(DATA_DIR + '/split/%s' % split, allow_pickle=True)
            df = df_loc_by_list(df, 'image_id', s)

        df['i'] = df['image_id'].map(lambda x: int(x.split('_')[-1]))
        df = df[['i', 'image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme']]

        self.uid = uid
        self.image = image
        self.df = df
        self.num_image = len(self.df)

        pass

    def __getitem__(self, index):
        i, image_id, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme = self.df.values[index]
        grapheme = TASK['grapheme'].class_map[grapheme]

        image = self.image[i].copy().reshape(137, 236)
        image = image.astype(np.float32) / 255
        label = [grapheme_root, vowel_diacritic, consonant_diacritic, grapheme]

        infor = Struct(
            index=index,
            image_id=image_id,
        )

        if self.augment is None:
            return image, label, infor
        else:
            return self.augment(image, label, infor)

    def __len__(self):
        return self.num_image

    def __str__(self):
        string = ''
        string += '\tlen = %d\n' % len(self)
        string += '\n'
        string += '\tmode     = %s\n' % self.mode
        string += '\tsplit    = %s\n' % self.split
        string += '\tcsv      = %s\n' % str(self.csv)
        string += '\tparquet  = %s\n' % self.parquet
        string += '\tnum_image = %d\n' % self.num_image
        return string


# see trorch/utils/data/sampler.py
class BalanceSampler(Sampler):
    def __init__(self, dataset, length):
        self.length = length

        df = dataset.df.reset_index()

        group = []
        grapheme_gb = df.groupby(['grapheme'])
        for k, i in TASK['grapheme'].class_map.items():
            g = grapheme_gb.get_group(k).index
            group.append(list(g))
            assert (len(g) > 0)

        self.group = group

    def __iter__(self):
        # l = iter(range(self.num_samples))
        # return l

        # for i in range(self.num_sample):
        #     yield i

        index = []
        n = 0

        is_loop = True
        while is_loop:
            num_class = TASK['grapheme'].num_class  # 1295
            c = np.arange(num_class)
            np.random.shuffle(c)
            for t in c:
                i = np.random.choice(self.group[t])
                index.append(i)
                n += 1
                if n == self.length:
                    is_loop = False
                    break
        return iter(index)

    def __len__(self):
        return self.length
