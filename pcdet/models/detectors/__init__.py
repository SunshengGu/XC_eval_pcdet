from .detector3d_template import Detector3DTemplate
from .second_net import SECONDNet
from .PartA2_net import PartA2Net
from .pv_rcnn import PVRCNN
from .pointpillar import PointPillar
# from .pointpillar2d import PointPillar2D


__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar
    # 'PointPillar2D': PointPillar2D
}


def build_detector(model_cfg, num_class, dataset, explain):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, explain=explain
    )

    return model
