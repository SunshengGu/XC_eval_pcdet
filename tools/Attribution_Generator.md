# AttributionGenerator
Author: Sunsheng Gu

`AttributionGenerator`, defined in `tools\attr_generator.py`, is a module that 
can compute pseudoimage attributions for PointPillars, as well as explanation 
concentration (XC), faraway attributions (far_attr), and pixel attribution 
prior loss (PAP, defined in the Expected Gradients paper).

## 1.0 What are supported?
### 1.1 Datasets
KITTI (tested)

CADC (not fully tested)
### 1.2 Models
PointPillars (tested)
### 1.3 Explanation Methods
From Captum:

IntegratedGradients (tested)

Saliency (tested)

GuidedGradCam (not tested)

## 2.0 Setup Instruction
### 2.1 Using the `XAI` branch
If you just pull the `XAI` branch, the `AttributionGenerator` would work, with only one extra step:
- Install Captum: `pip install captum`
### 2.2 Using your own branch
But if you need to work within a different branch, you may choose to merge your branch with `XAI`. 
Or you could do the following steps if solving merge conflicts is a big headache:
- Install Captum: `pip install captum`
- Go to our `WISEPCDet` repo, navigate to the `XAI` branch.
- Go to the `tools` folder, copy the `XAI_utils` folder and the `attr_generator.py` file into the `tools` folder of your local version of the `WISEPCDet` repo.
- Go to `pcdet/models/detectors/detector3d_template.py`, copy and paste the `post_processing_xai` into your `detector3d_template.py` file.
- Copy the following files into the corresponding directories in your local repo:
    - `pcdet/models/detectors/pointpillarxai.py`
    - `pcdet/models/backbones_3d/vfe/pillar_vfe_xai.py`
    - `pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter_xai.py`
    - `pcdet/models/backbones_2d/base_bev_backbone_xai.py`
    - `pcdet/models/dense_heads/anchor_head_single_xai.py`
    - `tools/cfgs/kitti_models/pointpillar_2DBackbone_DetHead_xai.yaml`
    - `tools/cfgs/kitti_models/pointpillar_xai.yaml`
    - `tools/cfgs/cadc_models/pointpillar_xai.yaml`
    - `tools/cfgs/cadc_models/pointpillar_2DBackbone_DetHead_xai.yaml`
- Then do the following:
    - For the folders where you added files ending with `...xai.py`, go to the `__init__.py` file in the corresponding folder.
    - Add the corresponding new modules into the `__init__.py` file.
    - Example:
        - In the `pcdet/models/backbones_2d` folder, you added the `base_bev_backbone_xai.py` file
        - Then in `pcdet/models/backbones_2d/__inin__.py`, you add `from .base_bev_backbone_xai import BaseBEVBackboneXAI` at the very beginning, and add `'BaseBEVBackboneXAI': BaseBEVBackboneXAI` in the big parenthesis following `__all__ =`.
    
## 3.0 How to Use
Go to `tools/attr_experiment.py` for an example of using the `AttributionGenerator`.

I chose to not feed the data_loader into the `AttributionGenerator`, because I think 
it would make more sense to have the data_loader running outside of `AttributionGenerator`.
As the data_loader is loading the data batch by batch, you can compute the acquisition
functions for this batch, as well as computing XC for this batch. So it works like this:
```buildoutcfg
for batch_index, batch in enumerate(data_loader):
    compute aquisition functions
    compute XC
    combine XC and aquisition function somehow
```

Hence, the `compute_xc` method in the `AttributionGenerator` only takes the batch_dict as input, along with two
other indicator arguments.

When calling the constructor, you can specify number of boxes per frame you want
to generate explanations for, whether to choose the most or least confident predictions
to explain and many more settings. See `attr_generator.py` for details.

Besides the constructor, the only 3 methods you should call are:
- `compute_xc`: Computes `XC`, `far_attr`, and `PAP`. Since you can get PAP almost for free, 
  so why not? Also, far_attr is already obtained in the process of computing XC, may
  as well get it too.
  
  When batch_size = 1, `XC` is a numpy array of shape (#_of_boxes_explained_per_frame).
  When batch_size > 1, `XC` is a numpy array of shape (batch_size, #_of_boxes_explained_per_frame)
  
  The same can be said for `far_attr`, and `PAP`.
- `compute_pap` : Computes PAP only, in case XC computation gets prohibitively slow.
- `reset`: Resets batch-wise parameters after processing a batch.

## 4.0 TODO
Vahdat & Nick:
- Save the computed XC, PAP, far_attr in a file specified by the `output_path` parameter

Sunsheng:
- Expand functionalities to accommodate batch_size > 1

