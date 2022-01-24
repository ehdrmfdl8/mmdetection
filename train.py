import mmcv
import os
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

if __name__ == '__main__':
    cfg = Config.fromfile('./configs/my_custom/mrcnn_config.py')
    cfg.work_dir = './mrcnn_dacon'
    cfg.gpu_ids = [0]
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)
    print(datasets)