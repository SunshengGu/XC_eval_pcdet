import csv
import glob
import os
import numpy as np

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_


def train_one_epoch(model, optimizer, train_loader, model_func, explained_model, explainer, attr_func, epoch_obj_cnt,
                    epoch_tp_obj_cnt, epoch_fp_obj_cnt, pred_score_file_name, pred_score_field_name, cur_epoch,
                    lr_scheduler, accumulated_iter, optim_cfg, rank, tbar, total_it_each_epoch, dataloader_iter,
                    cls_names, dataset_name, attr_loss, gt_infos, score_thresh, tb_log=None, leave_pbar=False,
                    box_selection="tp/fp"):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    # set to False in case we want verify something without altering the model
    enable_training = True
    constant_lr = False
    for cur_it in range(total_it_each_epoch):
        print("\nThe {}th batch\n".format(cur_it))
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        if not constant_lr:
            lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        try:
            beta1, beta2 = float(optimizer.betas[0]), float(optimizer.betas[1])
        except:
            beta1, beta2 = optimizer.param_groups[0]['betas'][0], optimizer.param_groups[0]['betas'][1]

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
        print("\nIn tools/train_utils/train_utils_new_loss.py, type(batch): {}".format(type(batch)))

        model.train()
        optimizer.zero_grad()

        # # Loading parameters from the training model to the explained model
        # explained_model.load_state_dict(model.state_dict(), strict=False)

        loss, tb_dict, disp_dict = model_func(model, batch)
        # print("\nin tools/train_utils/train_utils_new_loss.py, train_one_epoch, loss.shape {}".format(loss.shape)
        # print("loss data type is {}\n".format(type(loss)))
        if box_selection == "tp/fp":
            xc, far_attr, pap, fp_xc, fp_far_attr, fp_pap = attr_func(
                explained_model, explainer, batch, dataset_name, cls_names, cur_it=cur_it, gt_infos=gt_infos,
                score_thresh=score_thresh, object_cnt=epoch_obj_cnt, tp_object_cnt=epoch_tp_obj_cnt,
                fp_object_cnt=epoch_fp_obj_cnt, box_selection=box_selection, pred_score_file_name=pred_score_file_name,
                cur_epoch=cur_epoch, pred_score_field_name=pred_score_field_name
            )
        else:
            xc, far_attr, pap = attr_func(
                explained_model, explainer, batch, dataset_name, cls_names, cur_it=cur_it, gt_infos=gt_infos,
                score_thresh=score_thresh, object_cnt=epoch_obj_cnt, tp_object_cnt=epoch_tp_obj_cnt,
                fp_object_cnt=epoch_fp_obj_cnt, box_selection=box_selection, pred_score_file_name=pred_score_file_name,
                cur_epoch=cur_epoch, pred_score_field_name=pred_score_field_name
            )

        # print("xc_loss: {}".format(xc_loss))
        # print("far_attr: {}".format(far_attr))
        # print("pap: {}".format(pap))

        # need a scaling ratio to normalize the losses to the previous setting (top 3 per frame, batch_size = 2, taking
        # the average of frames in a batch)
        # TODO: handle the case when batch_box_cnt is zero
        batch_box_cnt = np.count_nonzero(~np.isnan(xc))
        scaling_ratio = batch["batch_size"] * 3 / batch_box_cnt if batch_box_cnt != 0 else 0
        xc_val = np.nansum(xc) / batch["batch_size"] * scaling_ratio
        pap_val = np.nansum(pap) / batch["batch_size"] * scaling_ratio
        far_attr_val = np.nansum(far_attr) / batch["batch_size"] * scaling_ratio
        xc_loss = 1 / xc_val if xc_val > 0.01 else 0 # smaller XC, bigger loss
        pap_loss = 0.001 * pap_val
        far_attr_loss = 0.03 * far_attr_val
        if box_selection == 'tp/fp':
            print("\ncomputing fp_xc_loss\n")
            fp_xc_loss = 0.1 * np.nansum(fp_xc) / batch["batch_size"]
        if attr_loss == 'xc' or attr_loss == 'XC':
            loss += xc_loss
            if box_selection == 'tp/fp':
                print("\nadding fp_xc_loss\n")
                loss += fp_xc_loss
        elif attr_loss == 'pap' or attr_loss == 'PAP':
            loss += pap_loss
        elif attr_loss == 'far_attr' or attr_loss == 'FAR_ATTR':
            loss += far_attr_loss
        # print("loss: {}".format(loss))
        if enable_training:
            loss.backward()
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('train/xc', xc_val, accumulated_iter)
                tb_log.add_scalar('train/xc_loss', xc_loss, accumulated_iter)
                if box_selection == 'tp/fp':
                    tb_log.add_scalar('train/fp_xc_loss', fp_xc_loss, accumulated_iter)
                tb_log.add_scalar('train/far_attr', far_attr_val, accumulated_iter)
                tb_log.add_scalar('train/far_attr_loss', far_attr_loss, accumulated_iter)
                tb_log.add_scalar('train/pap', pap_val, accumulated_iter)
                tb_log.add_scalar('train/pap_loss', pap_loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                tb_log.add_scalar('meta_data/beta1', beta1, accumulated_iter)
                tb_log.add_scalar('meta_data/beta2', beta1, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, logger, model_func, explained_model, explainer, attr_func,
                lr_scheduler, optim_cfg, start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, gt_infos,
                score_thresh, output_dir, train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, cls_names=['Car', 'Pedestrian', 'Cyclist'],
                dataset_name="KittiDataset", attr_loss="XC", box_selection="tp/fp"):
    # setup logging files for training-related boxes info
    obj_cnt_file_name = output_dir / 'interested_obj_cnt.csv'
    pred_score_file_name = output_dir / 'interested_pred_scores.csv'
    obj_cnt_field_name = ['epoch', 'tp_cnt', 'fp_cnt', 'tp_car_cnt', 'tp_pede_cnt', 'tp_cyc_cnt', 'fp_car_cnt',
                          'fp_pede_cnt', 'fp_cyc_cnt', 'car_cnt', 'pede_cnt', 'cyc_cnt']
    pred_score_field_name = ['epoch', 'batch', 'tp/fp', 'pred_label', 'pred_score']
    if box_selection != "tp/fp" and box_selection != "tp":
        obj_cnt_field_name = ['epoch', 'car_cnt', 'pede_cnt', 'cyc_cnt']
        pred_score_field_name = ['epoch', 'batch', 'pred_label', 'pred_score']
    with open(obj_cnt_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=obj_cnt_field_name)
        writer.writeheader()
    with open(pred_score_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=pred_score_field_name)
        writer.writeheader()
    accumulated_iter = start_iter
    if not (attr_loss == "XC" or attr_loss == "xc" or attr_loss == "PAP" or attr_loss == "pap" or
            attr_loss == "far_attr" or attr_loss == "FAR_ATTR" or attr_loss == "None"):
        raise NotImplementedError
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        epoch_obj_cnt = {}
        epoch_tp_obj_cnt = {}
        epoch_fp_obj_cnt = {}
        for i in range(len(cls_names)):
            epoch_obj_cnt[i] = 0
            epoch_tp_obj_cnt[i] = 0
            epoch_fp_obj_cnt[i] = 0
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)
            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func, explained_model, explainer, attr_func, epoch_obj_cnt,
                epoch_tp_obj_cnt, epoch_fp_obj_cnt,
                cur_epoch=cur_epoch,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                cls_names=cls_names, dataset_name=dataset_name, attr_loss=attr_loss, gt_infos=gt_infos,
                score_thresh=score_thresh, box_selection=box_selection, pred_score_file_name=pred_score_file_name,
                pred_score_field_name=pred_score_field_name
            )
            # log object count
            with open(obj_cnt_file_name, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=obj_cnt_field_name)
                data_dict = {}
                data_dict["epoch"] = cur_epoch
                #TODO: fill the data dict
                if box_selection == "tp/fp" or box_selection == "tp":
                    data_dict["tp_car_cnt"], data_dict["fp_car_cnt"] = epoch_tp_obj_cnt[0], epoch_fp_obj_cnt[0]
                    data_dict["tp_pede_cnt"], data_dict["fp_pede_cnt"] = epoch_tp_obj_cnt[1], epoch_fp_obj_cnt[1]
                    data_dict["tp_cyc_cnt"], data_dict["fp_cyc_cnt"] = epoch_tp_obj_cnt[2], epoch_fp_obj_cnt[2]
                    data_dict["tp_cnt"], data_dict["fp_cnt"] = sum(epoch_tp_obj_cnt.values()), sum(epoch_fp_obj_cnt.values())
                data_dict["car_cnt"], data_dict["pede_cnt"], data_dict["cyc_cnt"] = epoch_obj_cnt[0], epoch_obj_cnt[1], epoch_obj_cnt[2]
                writer.writerow(data_dict)
            logger.info("{}:{} {}:{} {}:{} after training for {} epochs".format(
                cls_names[0], epoch_obj_cnt[0], cls_names[1], epoch_obj_cnt[1], cls_names[2], epoch_obj_cnt[2],
                cur_epoch))
            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
