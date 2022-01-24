import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset
from glob import glob
import json
from collections import OrderedDict

def get_data_paths(dataroot):
    paths = None
    if dataroot is not None:
        paths = sorted(glob(dataroot+'/*'))
        paths = [path.replace('\\','/') for path in paths]
    return paths

def parse(path):
    json_str = ''
    with open(path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    return opt

Disease = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']
@DATASETS.register_module()
class DaconDataset(CustomDataset):

    CLASSES = ('00', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8')

    def load_annotations(self, ann_file):
        ann_list = get_data_paths(ann_file)

        data_infos = []
        disease_encoder = {key: idx for idx, key in enumerate(self.CLASSES)}
        for i, ann_path in enumerate(ann_list):
            file_name = ann_path.split('/')[-1]
            ann_path = f'{ann_path}/{file_name}.json'
            ann_line = parse(ann_path)
            # if ann_line != '#':
            #     continue
            width = ann_line['description']['width']
            height = ann_line['description']['height']
            crop = ann_line['annotations']['crop']
            area = ann_line['annotations']['area']
            disease = disease_encoder[ann_line['annotations']['disease']]
            risk = ann_line['annotations']['risk']
            bboxes = []
            labels = []
            if ann_line['annotations']['part'] == []:
                bbox = [0.0,0.0,0.0,0.0]
                # bbox.append(ann_line['annotations']['bbox'][0]['x'])
                # bbox.append(ann_line['annotations']['bbox'][0]['y'])
                # bbox.append(ann_line['annotations']['bbox'][0]['w'])
                # bbox.append(ann_line['annotations']['bbox'][0]['h'])
                bboxes.append(bbox)
                labels.append(disease)
            else:
                for disease_bbox in ann_line['annotations']['part']:
                    bbox = []
                    bbox.append(disease_bbox['x'])
                    bbox.append(disease_bbox['y'])
                    bbox.append(disease_bbox['w'])
                    bbox.append(disease_bbox['h'])
                    bboxes.append(bbox)
                    labels.append(disease)


            data_infos.append(
                dict(
                    filename=file_name,
                    width=width,
                    height=height,
                    ann=dict(
                        labels=np.array(labels).astype(np.int64),
                        bboxes=np.array(bboxes).astype(np.float32)
                        )
                ))

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

