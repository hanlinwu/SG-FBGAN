#! /usr/bin/python
# -*- coding: utf8 -*-
import tensorflow as tf
from utils.get_model import GetModel
from data_loader.load_data import Data
from utils.config import load_config, log_config
from utils.utils import get_args

import time,os

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
    except:
        print("missing or invalid arguments")
        exit(0)

    exp_name = args.opt
    args.opt = 'experiments/{0}/config.json'.format(exp_name)
    config = load_config(args.opt, is_train = False, exp_name= exp_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not config.exp_name :
        raise Exception('please specify experiment name')

    # create the data generator
    data = Data(config)

    # create an instance of the model
    Model = GetModel(config.model)
    model = Model(config)

    # create trainer and pass all the previous components to it
    model.compile(data)

    #load model if exists
    model.load()

    # evaluate the model
    model.evaluate()

if __name__ == '__main__':
    main()