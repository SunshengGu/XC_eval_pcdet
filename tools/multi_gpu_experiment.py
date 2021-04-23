import os
import numpy as np
from attr_generator_mult import *
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch import Tensor
from torch.multiprocessing import Process

USE_CUDA = True
WORLD_SIZE = 2
MULTI = True


def run(rank, inp_batch, XCCalculator):
    # Move model and input to device with ID rank if USE_CUDA is True
    if USE_CUDA:
        inp_batch = inp_batch.cuda(rank)
        XCCalculator.move_model(rank)

    # Combine attributions from each device using distributed.gather
    # Rank 0 process gathers all attributions, each other process
    # sends its corresponding attribution.
    if rank == 0:
        batch_XC, batch_far_attr, batch_total_pap = XCCalculator.compute_xc(inp_batch, method="sum")
        print("\nXC values for the batch:\n {}".format(batch_XC))
        XCCalculator.reset()
    else:
        XCCalculator.compute_xc(inp_batch, method="sum")


def init_process(rank, fn, inp_batch, XCCalculator, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, inp_batch, XCCalculator)
    dist.destroy_process_group()


def main():
    '''________________________User Input Begin________________________'''
    dataset_name = 'KittiDataset'  # change to your dataset, follow naming convention in the config yaml files in PCDet
    method = 'IntegratedGradients'  # explanation method
    attr_shown = 'positive'  # show positive or negative attributions

    # IG specific parameters
    mult_by_inputs = True  # whether to show attributions only at where some input exists
    steps = 24  # number of intermediate steps for IG

    # config file for the full PointPillars model
    cfg_file = 'cfgs/kitti_models/pointpillar_xai.yaml'
    # config file for the truncated PointPillars model: 2D backbone and SSD
    explained_cfg_file = 'cfgs/kitti_models/pointpillar_2DBackbone_DetHead_xai.yaml'

    # model checkpoint, change to your checkpoint, make sure it's a well-trained model
    ckpt = '../output/kitti_models/pointpillar/default/ckpt/pointpillar_7728.pth'

    num_batchs = 24

    '''________________________User Input End________________________'''
    # data set prepration, use the validation set (called 'test_set' here)
    # arguments for dataloader
    batch_size = 2
    workers = 0
    dist_test = False
    cfg_from_yaml_file(explained_cfg_file, cfg)
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=dist_test, workers=workers, training=False
    )

    myXCCalculator = AttributionGenerator(model_ckpt=ckpt, full_model_cfg_file=cfg_file,
                                          model_cfg_file=explained_cfg_file,
                                          data_set=test_set, xai_method=method, output_path="unspecified",
                                          ignore_thresh=0.0, debug=True, multi_gpu=MULTI, world_size=WORLD_SIZE)

    for batch_num, batch_dictionary in enumerate(test_loader):
        if batch_num == num_batchs:
            break
        print("\nAnalyzing the {}th batch".format(batch_num))
        batch_chunks = batch_dictionary.chunk(WORLD_SIZE)
        processes = []
        for rank in range(WORLD_SIZE):
            p = Process(target=init_process, args=(rank, run, batch_chunks[rank], myXCCalculator))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == '__main__':
    main()
