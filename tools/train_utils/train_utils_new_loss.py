import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_


def train_one_epoch(model, optimizer, train_loader, model_func, explained_model, explainer, attr_func,
                    lr_scheduler, accumulated_iter, optim_cfg, rank, tbar, total_it_each_epoch, dataloader_iter,
                    cls_names, dataset_name, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        print("\nThe {}th batch\n".format(cur_it))
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

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
        pap_only = True
        if not pap_only:
            xc, xc_loss, far_attr, pap = attr_func(explained_model, explainer, batch, dataset_name, cls_names)
        else:
            pap = attr_func(explained_model, explainer, batch, dataset_name, cls_names, pap_only=True)
        # print("xc_loss: {}".format(xc_loss))
        # print("far_attr: {}".format(far_attr))
        # print("pap: {}".format(pap))
        pap_loss = 0.001 * pap
        if not pap_only:
            loss += xc_loss
        else:
            loss += pap_loss
        # print("loss: {}".format(loss))
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
                if not pap_only:
                    tb_log.add_scalar('train/xc', xc, accumulated_iter)
                    tb_log.add_scalar('train/xc_loss', xc_loss, accumulated_iter)
                    tb_log.add_scalar('train/far_attr', far_attr, accumulated_iter)
                tb_log.add_scalar('train/pap_loss', pap_loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, explained_model, explainer, attr_func,
                lr_scheduler, optim_cfg, start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, cls_names=['Car', 'Pedestrian', 'Cyclist'],
                dataset_name="KittiDataset"):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func, explained_model, explainer, attr_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                cls_names=cls_names, dataset_name=dataset_name
            )

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
