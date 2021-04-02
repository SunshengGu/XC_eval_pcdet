from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .anchor_head_single_new_loss import AnchorHeadSingleNewLoss
from .anchor_head_template_new_loss import AnchorHeadTemplateNewLoss
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead

from .anchor_head_single_xai import AnchorHeadSingleXAI

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadTemplateNewLoss': AnchorHeadTemplateNewLoss,
    'AnchorHeadSingle': AnchorHeadSingle,
    'AnchorHeadSingleXAI': AnchorHeadSingleXAI,
    'AnchorHeadSingleNewLoss': AnchorHeadSingleNewLoss,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
}
