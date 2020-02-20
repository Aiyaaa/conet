from os import path

class Cfg:
    # self.data_dir = '/data2/hangli_data/oct/data/2015_BOE_Chiu'
    def __init__(self):
        super().__init__()
        base_dir = '/data1/hangli'
        self.s_data_dir =     path.join(base_dir, 'oct/data/2015_BOE_Chiu')
        self.data_dir =       path.join(base_dir, 'oct/data/converted')
        self.data_sp_dir =    path.join(base_dir, 'oct/data/superpixel')
        self.data_gt_dir =    path.join(base_dir, 'oct/data/converted/gt')
        self.dme_flatten =    path.join(base_dir, 'oct/data/dme_flatten')
        self.dme_flatten_sp = path.join(base_dir, 'oct/data/dme_flatten_sp')

        self.hc =          path.join(base_dir, 'oct/data/hc')
        self.hc_train =    path.join(base_dir, 'oct/data/hc_train')
        self.hc_test =     path.join(base_dir, 'oct/data/hc_test')
        self.hc_train_sp = path.join(base_dir, 'oct/data/hc_train_sp')
        self.hc_test_sp =  path.join(base_dir, 'oct/data/hc_test_sp')
        self.hc_sp =       path.join(base_dir, 'oct/data/hc_sp')


def get_cfg():
    cfg = Cfg()
    return cfg