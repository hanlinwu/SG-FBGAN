import os, json, time
from easydict import EasyDict as edict
from dirs import create_dirs
import copy

def load_config(file_dir, is_train, exp_name="E"):
    with open(file_dir) as f:
        config = json.load(f)
    config = edict(config)
    data_path = config.data_path

    config.is_train = is_train
    if hasattr(config.TRAIN, 'hr_img_path') and hasattr(config.TRAIN, 'hr_mid_path') : 
        config.TRAIN.hr_img_path = os.path.join(data_path, config.TRAIN.hr_img_path)
        config.TRAIN.hr_saliency_mask = os.path.join(data_path, config.TRAIN.hr_saliency_mask)
        config.TRAIN.hr_saliency_map = os.path.join(data_path, config.TRAIN.hr_saliency_map)
        config.TRAIN.hr_mid_path = os.path.join(data_path, config.TRAIN.hr_mid_path)
        config.VALID.hr_img_path = os.path.join(data_path, config.VALID.hr_img_path)
        config.VALID.hr_saliency_mask = os.path.join(data_path, config.VALID.hr_saliency_mask)
        config.VALID.hr_saliency_map = os.path.join(data_path, config.VALID.hr_saliency_map)
        config.VALID.hr_mid_path = os.path.join(data_path, config.VALID.hr_mid_path)

    if config.is_train and config.exp_name and config.is_continue_train == False :
        raise Exception('when config.exp_name is specified, config.is_continue_train mast be True')
    
    if not config.exp_name :
        config.exp_name = exp_name + time.strftime("_%Y%m%dT%H%M%S", time.localtime())
    
    config.experiment_dir = os.path.join('experiments', config.exp_name)
    config.checkpoint_dir = os.path.join(config.experiment_dir, 'checkpoint')
    config.summary_dir = os.path.join(config.experiment_dir, 'summary')
    config.evaluate_dir = os.path.join(config.experiment_dir, 'evaluate')

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.evaluate_dir])
    
    # some default settings...
    if not hasattr(config.MODEL, 'D') : config.MODEL.D = 2
    if not hasattr(config.MODEL, 'C') : config.MODEL.C = 3
    if not hasattr(config.MODEL, 'degradation_model') : config.MODEL.degradation_model = "BI"
    
    if config.is_train:
        log_config(os.path.join(config.experiment_dir, 'config.json'), config)

    return config

def log_config(filename, cfg):
    cfg_c = copy.deepcopy(cfg)
    data_path = cfg.data_path
    if hasattr(cfg_c.TRAIN, 'hr_img_path') : 
        cfg_c.TRAIN.hr_img_path = cfg.TRAIN.hr_img_path.replace(data_path,'')
        cfg_c.TRAIN.hr_saliency_mask = cfg.TRAIN.hr_saliency_mask.replace(data_path,'')
        cfg_c.TRAIN.hr_saliency_map = cfg.TRAIN.hr_saliency_map.replace(data_path,'')
        cfg_c.VALID.hr_img_path = cfg.VALID.hr_img_path.replace(data_path,'')
        cfg_c.VALID.hr_saliency_mask = cfg.VALID.hr_saliency_mask.replace(data_path,'')
        cfg_c.VALID.hr_saliency_map = cfg.VALID.hr_saliency_map.replace(data_path,'')
    cfg_c.pop('checkpoint_dir')
    cfg_c.pop("summary_dir")
    cfg_c.pop("experiment_dir")
    cfg_c.pop("evaluate_dir")
    cfg_c.pop("is_train")
    with open(filename, 'w') as f:
        f.write(json.dumps(cfg_c, indent=4))