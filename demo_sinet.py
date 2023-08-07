# encoding: utf-8
# !/usr/bin/env python3
import torch
import matplotlib as mpl
import os
import yaml
import shutil
import pickle
import argparse
import datetime
import numpy as np
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG
from sklearn.model_selection import train_test_split

torch.backends.cudnn.benchmark = True

from model.sinet import SINet  # noqa
import util.util_dataloader as util_dataloader  # noqa

mpl.use('Agg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--expid', type=int, default=0)
    parser.add_argument('--config', help="configuration file *.yml",
                        type=str, required=False, default='./config/GWN-GWN-MLP/template.yml')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()

    # load yaml setting file
    args_dict = vars(args)
    cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args_dict.update(cfg)
    print("arguments: {}".format(str(args)))

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    np.random.seed(1)
    torch.manual_seed(1)
    # -------------------------------- #
    dirpath_new = './data/'

    traintestpath = f'{dirpath_new}/experiment/dataset_{args.expid}/traintest_split_id/'

    args.dirpath = dirpath_new
    args.savepath = f'./out/dataset_{args.expid}'

    # pytorch logger
    logname = '{}_{}'.format(
        args.model, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
    args.log_dir = f'{args.savepath}/runs/{args.model}/{logname}'

    # -------------------------------- #
    for i in ['runs']:
        path = f'{args.savepath}/{i}/'
        if not os.path.exists(path):
            os.mkdir(path)

    os.makedirs(args.log_dir)
    # -------------------------------- #

    # -------------------------------- ##
    # Logger
    logger = getLogger("Pytorch")
    logger.setLevel(DEBUG)
    handler_format = Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(handler_format)

    file_handler = FileHandler(f'{args.log_dir}/{logname}.log', 'a')

    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.debug("Start training.")
    logger.debug("SummaryWriter outputs to %s" % (args.log_dir))

    shutil.copyfile(args.config, f'{args.log_dir}/config.yml')

    # -------------------------------- #

    _train_id = np.loadtxt('%s/prop_train_id.csv' % (traintestpath))
    test_id = np.loadtxt('%s/prop_test_id.csv' % (traintestpath))
    train_id, valid_id = train_test_split(
        _train_id, random_state=123, test_size=1-args.trainprop)

    train_dataset = util_dataloader.ShinkokuDataset(
        id=train_id, Nguide=args.guide, mode='train', expid=args.expid, args=args)

    # Global Graph
    G = train_dataset.graph.get()
    # Local Graph
    A = train_dataset.getseatgraph.get_graph()

    y_scaler = pickle.load(open(dirpath+'data/y_scaler.pkl', 'rb'))
    model = SINet(args.din, args.dtreat, args.dout,
                  G, A, y_scaler, args).to(device=args.device)
    id, log_dir, require_test = model.check_json(False)

    if id != None:
        args.log_dir = log_dir
        model.args.log_dir = log_dir
        model.load_state_dict(torch.load(f'{args.log_dir}/model.pt'))
    else:
        valid_dataset = util_dataloader.ShinkokuDataset(
            id=valid_id, Nguide=args.guide, mode='valid', expid=args.expid)

        test_dataset_cs = util_dataloader.ShinkokuDataset(
            id=test_id, Nguide=args.guide, mode='train', expid=args.expid)

        in_dataset = util_dataloader.ShinkokuDataset(
            id=train_id, Nguide=args.guide, mode='test', expid=args.expid)
        out_dataset = util_dataloader.ShinkokuDataset(
            id=test_id, Nguide=args.guide, mode='test', expid=args.expid)

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
        validloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=False)
        testloader_cs = torch.utils.data.DataLoader(
            test_dataset_cs, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=False)
        inloader = torch.utils.data.DataLoader(
            in_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
        outloader = torch.utils.data.DataLoader(
            out_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

        if args.device == 'cuda':
            model = torch.nn.DataParallel(model)  # make parallel
            torch.cudnn.benchmark = True
        logger.debug('Model Structure.')
        logger.debug(args)
        logger.debug(model)
        losses = model.fit(
            trainloader, validloader, inloader, outloader, testloader_cs)

        torch.save(model.state_dict(),
                   f'{model.args.log_dir}/model.pt')

    # Test
    if False:  # require_test:
        in_dataset = util_dataloader.ShinkokuDataset(
            id=train_id, Nguide=args.guide, mode='test', expid=args.expid)
        out_dataset = util_dataloader.ShinkokuDataset(
            id=test_id, Nguide=args.guide, mode='test', expid=args.expid)
        inloader = torch.utils.data.DataLoader(
            in_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
        outloader = torch.utils.data.DataLoader(
            out_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

        model.test(inloader, outloader)

    logger.debug(0)
