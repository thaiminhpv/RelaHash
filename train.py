import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pprint import pprint

import torch
from torch.optim import Adam

import configs
from hashing.utils import calculate_accuracy, get_hamm_dist, calculate_mAP
from networks.loss import RelaHashLoss
from networks.model import RelaHash
from utils import io
from utils.misc import AverageMeter, Timer


def train_hashing(optimizer, model, centroids, train_loader, loss_param):
    model.train()
    device = loss_param['device']
    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()

    total_timer.tick()

    criterion = RelaHashLoss(**loss_param)

    for i, (data, labels) in enumerate(train_loader):
        timer.tick()

        # clear gradient
        optimizer.zero_grad()

        data, labels = data.to(device), labels.to(device)
        logits, codes = model(data)

        loss = criterion(logits, codes, labels)

        # backward and update
        loss.backward()
        optimizer.step()

        hamm_dist = get_hamm_dist(codes, centroids, normalize=True)
        acc, cbacc = calculate_accuracy(logits, hamm_dist, labels, loss_param['multiclass'])

        timer.toc()
        total_timer.toc()

        # store results
        meters['loss_total'].update(loss.item(), data.size(0))
        meters['acc'].update(acc.item(), data.size(0))
        meters['cbacc'].update(cbacc.item(), data.size(0))

        meters['time'].update(timer.total)

        print(f'Train [{i + 1}/{len(train_loader)}] '
              f'T: {meters["loss_total"].avg:.4f} '
              f'A(CE): {meters["acc"].avg:.4f} '
              f'A(CB): {meters["cbacc"].avg:.4f} '
              f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')

    print()
    total_timer.toc()

    meters['total_time'].update(total_timer.total)

    return meters


def test_hashing(model, centroids, test_loader, loss_param, return_codes=False):
    model.eval()
    device = loss_param['device']
    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()

    total_timer.tick()

    ret_codes = []
    ret_labels = []

    criterion = RelaHashLoss(**loss_param)

    for i, (data, labels) in enumerate(test_loader):
        timer.tick()

        with torch.no_grad():
            data, labels = data.to(device), labels.to(device)
            logits, codes = model(data)

            loss = criterion(logits, codes, labels)

            hamm_dist = get_hamm_dist(codes, centroids, normalize=True)
            acc, cbacc = calculate_accuracy(logits, hamm_dist, labels, loss_param['multiclass'])

            if return_codes:
                ret_codes.append(codes)
                ret_labels.append(labels)

        timer.toc()
        total_timer.toc()

        # store results
        meters['loss_total'].update(loss.item(), data.size(0))
        meters['acc'].update(acc.item(), data.size(0))
        meters['cbacc'].update(cbacc.item(), data.size(0))

        meters['time'].update(timer.total)

        print(f'Test [{i + 1}/{len(test_loader)}] '
              f'T: {meters["loss_total"].avg:.4f} '
              f'A(CE): {meters["acc"].avg:.4f} '
              f'A(CB): {meters["cbacc"].avg:.4f} '
              f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')

    print()
    meters['total_time'].update(total_timer.total)

    if return_codes:
        res = {
            'codes': torch.cat(ret_codes),
            'labels': torch.cat(ret_labels)
        }
        return meters, res

    return meters


def prepare_dataloader(config):
    logging.info('Creating Datasets')
    train_dataset = configs.dataset(config, filename='train.txt', transform_mode='train')

    separate_multiclass = config['dataset_kwargs'].get('separate_multiclass', False)
    config['dataset_kwargs']['separate_multiclass'] = False
    test_dataset = configs.dataset(config, filename='test.txt', transform_mode='test')
    db_dataset = configs.dataset(config, filename='database.txt', transform_mode='test')
    config['dataset_kwargs']['separate_multiclass'] = separate_multiclass  # during mAP, no need to separate

    logging.info(f'Number of DB data: {len(db_dataset)}')
    logging.info(f'Number of Train data: {len(train_dataset)}')

    train_loader = configs.dataloader(train_dataset, config['batch_size'])
    test_loader = configs.dataloader(test_dataset, config['batch_size'], shuffle=False, drop_last=False)
    db_loader = configs.dataloader(db_dataset, config['batch_size'], shuffle=False, drop_last=False)

    return train_loader, test_loader, db_loader


def main(config):
    device = torch.device(config.get('device', 'cuda:0'))

    io.init_save_queue()

    start_time = time.time()
    configs.seeding(config['seed'])

    logdir = config['logdir']
    assert logdir != '', 'please input logdir'

    pprint(config)

    if config['wandb_enable']:
        import wandb
        ## initiaze wandb ##
        wandb_dir = logdir
        wandb.init(project="relahash", config=config, dir=wandb_dir)
        # wandb run name
        wandb.run.name = logdir.split('logs/')[1]


    os.makedirs(f'{logdir}/models', exist_ok=True)
    os.makedirs(f'{logdir}/optims', exist_ok=True)
    os.makedirs(f'{logdir}/outputs', exist_ok=True)
    json.dump(config, open(f'{logdir}/config.json', 'w+'), indent=4, sort_keys=True)

    nclass = config['arch_kwargs']['nclass']
    nbit = config['arch_kwargs']['nbit']

    train_loader, test_loader, db_loader = prepare_dataloader(config)
    model = RelaHash(**config['arch_kwargs'])
    model.to(device)
    print(model)

    logging.info(f'Total Bit: {nbit}')
    centroids = model.get_centroids()
    io.fast_save(centroids, f'{logdir}/outputs/centroids.pth')

    if config['wandb_enable']:
        wandb.watch(model)

    backbone_lr_scale = 0.1
    optimizer = Adam([
            {'params': model.get_backbone_params(), 'lr': config['optim_kwargs']['lr'] * backbone_lr_scale},
            {'params': model.get_hash_params()}
        ],
        lr=config['optim_kwargs']['lr'],
        betas=config['optim_kwargs'].get('betas', (0.9, 0.999)),
        weight_decay=config['optim_kwargs'].get('weight_decay', 0))
    scheduler = configs.scheduler(config, optimizer)

    train_history = []
    test_history = []

    loss_param = config.copy()
    loss_param.update({'device': device})

    best = 0
    curr_metric = 0

    nepochs = config['epochs']
    neval = config['eval_interval']

    logging.info('Training Start')

    for ep in range(nepochs):
        logging.info(f'Epoch [{ep + 1}/{nepochs}]')
        res = {'ep': ep + 1}

        train_meters = train_hashing(optimizer, model, centroids, train_loader, loss_param)
        scheduler.step()

        for key in train_meters: res['train_' + key] = train_meters[key].avg
        train_history.append(res)
        # train_outputs.append(train_out)
        if config['wandb_enable']:
            wandb_train = res.copy()
            wandb_train.pop("ep")
            wandb.log(wandb_train, step=res['ep'])


        eval_now = (ep + 1) == nepochs or (neval != 0 and (ep + 1) % neval == 0)
        if eval_now:
            res = {'ep': ep + 1}

            test_meters, test_out = test_hashing(model, centroids, test_loader, loss_param, True)
            db_meters, db_out = test_hashing(model, centroids, db_loader, loss_param, True)

            for key in test_meters: res['test_' + key] = test_meters[key].avg
            for key in db_meters: res['db_' + key] = db_meters[key].avg

            res['mAP'] = calculate_mAP(db_out['codes'], db_out['labels'],
                                       test_out['codes'], test_out['labels'],
                                       loss_param['R'])
            logging.info(f'mAP: {res["mAP"]:.6f}')

            curr_metric = res['mAP']
            test_history.append(res)
            # test_outputs.append(outs)

            if config['wandb_enable']:
                wandb_test = res.copy()
                wandb_test.pop("ep")
                wandb.log(wandb_test, step=res['ep'])
            if best < curr_metric:
                best = curr_metric
                io.fast_save(modelsd, f'{logdir}/models/best.pth')
                if config['wandb_enable']:
                    wandb.run.summary["best_map"] = best



        json.dump(train_history, open(f'{logdir}/train_history.json', 'w+'), indent=True, sort_keys=True)
        # io.fast_save(train_outputs, f'{logdir}/outputs/train_last.pth')

        if len(test_history) != 0:
            json.dump(test_history, open(f'{logdir}/test_history.json', 'w+'), indent=True, sort_keys=True)
            # io.fast_save(test_outputs, f'{logdir}/outputs/test_last.pth')

        modelsd = model.state_dict()
        # optimsd = optimizer.state_dict()
        # io.fast_save(modelsd, f'{logdir}/models/last.pth')
        # io.fast_save(optimsd, f'{logdir}/optims/last.pth')
        save_now = config['save_interval'] != 0 and (ep + 1) % config['save_interval'] == 0
        if save_now:
            io.fast_save(modelsd, f'{logdir}/models/ep{ep + 1}.pth')
            # io.fast_save(optimsd, f'{logdir}/optims/ep{ep + 1}.pth')
            # io.fast_save(train_outputs, f'{logdir}/outputs/train_ep{ep + 1}.pth')

        if best < curr_metric:
            best = curr_metric
            io.fast_save(modelsd, f'{logdir}/models/best.pth')

    modelsd = model.state_dict()
    io.fast_save(modelsd, f'{logdir}/models/last.pth')
    total_time = time.time() - start_time
    io.join_save_queue()
    logging.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
    logging.info(f'Best mAP: {best:.6f}')
    logging.info(f'Done: {logdir}')

    return logdir
