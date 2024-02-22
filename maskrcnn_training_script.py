import json
import detectron2
import os
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from detectron2.utils.visualizer import Visualizer
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.utils.visualizer import ColorMode
from tqdm.notebook import tqdm
import matplotlib.patches as patches
import warnings
from detectron2.data import detection_utils as utils
from skimage import measure
import detectron2.data.transforms as T
import glob
import yaml
import copy
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from osgeo import gdal
import albumentations as A
warnings.filterwarnings("ignore")

def get_image(raster_path):
    def norma_data(data, norma_methods="dw"):
        arr = np.empty(data.shape, dtype=np.float32)
        for i in range(data.shape[-1]):
            array = data[:, :, i]
            mi_1, ma_99, mi_30, ma_70 = np.percentile(array, 1), np.percentile(array, 99), \
                                        np.percentile(array, 30), np.percentile(array, 70)
            if norma_methods == "dw":
                new_array = np.log(array * 0.0001 + 1)
                new_array = (new_array - mi_30 * 0.0001) / (ma_70 * 0.0001)
                new_array = np.exp(new_array * 5 - 1)
                new_array = new_array / (new_array + 1)

            else:
                new_array = (1*(array-mi_1)/(ma_99-mi_1)).clip(0, 1)
            arr[:, :, i] = new_array
        return arr

    ds = gdal.Open(raster_path)
    image = np.empty((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=np.float32)
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b).ReadAsArray()
        image[:, :, b-1] = band
    if image.shape[-1] == 1:
        image = image[:, :, 0]
    else:
        image = norma_data(image, norma_methods='min-max')
    return image

detectron2.data.detection_utils.read_image = get_image

def get_tree_dicts(indices, annotation):
    img_dir=MULTICLASS_IMAGES_DIR
    
    convert_xywh_to_xyxy = lambda xywh_rel: [xywh_rel[0], xywh_rel[1], xywh_rel[0] + xywh_rel[2], xywh_rel[1] + xywh_rel[3]]
    
    dataset_dicts = []
    
    df1 = pd.DataFrame(annotation['images'])
    images = df1[df1.id.isin(indices)]
    df2 = pd.DataFrame(annotation['annotations'])
    annots = df2[df2.image_id.isin(indices)]
    mask_obj = [j for _,j in annots.groupby('image_id')]
    
    images = images.to_dict(orient='records')
        
    for image, annot in zip(images, mask_obj):
        record = {}
        
        filename = os.path.join(img_dir, image['file_name'])
        height, width = image['width'], image['height']
        
        record["file_name"] = filename
        record["image_id"] = image['id']
        record["height"] = height
        record["width"] = width
        
        objs = []
        for i in range(annot.shape[0]):
            ann = annot.iloc[i]
            # print(sm[ann['category_id']])
            o = {
                "bbox": convert_xywh_to_xyxy(ann['bbox']),
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": ann['segmentation'],
                "category_id": sm[ann['category_id']],
                'area': ann['area'],
                'iscrowd': ann['iscrowd']
            }
            objs.append(o)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
        
    return dataset_dicts

def set_config(train_means, train_stds):    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("tree_train",)
    cfg.DATASETS.TEST = ("tree_val",)
    cfg.DATALOADER.NUM_WORKERS = os.cpu_count()
    cfg.INPUT.FORMAT = 'RGB'
    cfg.MODEL.PIXEL_MEAN = train_means.mean(axis=0).tolist()
    cfg.MODEL.PIXEL_STD = train_stds.mean(axis=0).tolist()
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 20000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13+1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR = './maskrcnn_output'
    
    return cfg
    
def setup(ti=None,vi=None):
    if ti == None and vi == None:
        image_ids = [img['id'] for img in obj['images']]
        random.shuffle(image_ids)
        split_idx = len(image_ids) * 75 // 100
        ti = image_ids[:split_idx]
        vi = image_ids[split_idx:]

    train_tree_dicts = get_tree_dicts(indices=ti,annotation=obj)
    # val_tree_dicts = get_tree_dicts(indices=val_indices,annotation=obj)

    DatasetCatalog.register("tree_train", lambda: get_tree_dicts(indices=ti, annotation=obj))
    MetadataCatalog.get("tree_train").set(thing_classes=list(map(str,list(sm.values())[1:])))

    # Register validation dataset
    DatasetCatalog.register("tree_val", lambda: get_tree_dicts(indices=vi, annotation=obj))
    MetadataCatalog.get("tree_val").set(thing_classes=list(map(str,list(sm.values())[1:])))

    train_means, train_stds = [],[]
    for tree_dict in tqdm(train_tree_dicts):
        img = utils.read_image(tree_dict['file_name'])
        train_means.append(np.mean(img, axis=(0,1)))
        train_stds.append(np.std(img, axis=(0,1)))
    
    train_means = np.array(train_means)
    train_stds = np.array(train_stds)

    cfg = set_config(train_means=train_means, train_stds=train_stds)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg, ti, vi

def run_training(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def run_inference(cfg, val_tree_dicts):
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    # f"./{cfg.OUTPUT_DIR}/model_final.pth"
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("tree_val", cfg, False, output_dir="./maskrcnn_output/")
    val_loader = build_detection_test_loader(cfg, "tree_val")
    return inference_on_dataset(predictor.model, val_loader, evaluator)

    # val_seg_masks = []

    # for i in tqdm(range(len(val_tree_dicts))):
    #     im = utils.read_image(val_tree_dicts[i]["file_name"])
    #     # index = val_tree_dicts[i]["file_name"].split('_')[-1].rstrip('.tif')
    #     # mask = get_image([x for x in new_multicls_masks if index in x][0])

    #     outputs = predictor(im[...,:3])


if __name__ == "__main__":
    with open('./maskrcnn_config.yaml','r') as f:
        CONFIG = yaml.safe_load(f)
    
    ANNOTATIONS_PATH = CONFIG['ANNOTATIONS_PATH']
    SPECIES_MAPPING_PATH = CONFIG['SPECIES_MAPPING_PATH']
    MULTICLASS_IMAGES_DIR = CONFIG['MULTICLASS_IMAGES_DIR']

    with open(ANNOTATIONS_PATH, 'r') as f:
        obj = json.load(f)

    with open(SPECIES_MAPPING_PATH,'r') as f:
        species_mapping = f.read()
        sm = {}
        sm[0] = 0
        
        for idx, i in enumerate(species_mapping.split('\n')[2:-1]):
            sm[int(i)] = idx
    
    cfg, training_indices, validation_indices = setup()
    run_training(cfg=cfg)
