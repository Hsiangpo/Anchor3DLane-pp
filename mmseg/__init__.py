# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
from packaging.version import parse

from .version import __version__, version_info

# 兼容 NumPy 2.x：旧代码仍使用 np.float/np.int/np.bool 这些别名。
for _name, _value in {
        'float': float,
        'int': int,
        'bool': bool,
        'object': object,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)


def _patch_mmcv_scatter_get_stream():
    # 兼容 torch 2.x：旧版 mmcv scatter 会把整数 GPU id 传给 _get_stream。
    try:
        from mmcv.parallel import _functions as mmcv_parallel_functions
    except Exception:
        return

    original_get_stream = mmcv_parallel_functions._get_stream
    if getattr(original_get_stream, '_anchor3dlane_compat', False):
        return

    def get_stream_compat(device):
        if isinstance(device, int):
            device = torch.device('cuda', device)
        return original_get_stream(device)

    get_stream_compat._anchor3dlane_compat = True
    mmcv_parallel_functions._get_stream = get_stream_compat


_patch_mmcv_scatter_get_stream()

MMCV_MIN = '1.3.13'
MMCV_MAX = '1.7.2'


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


mmcv_min_version = digit_version(MMCV_MIN)
mmcv_max_version = digit_version(MMCV_MAX)
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_min_version <= mmcv_version < mmcv_max_version), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_min_version}, <{mmcv_max_version}.'

__all__ = ['__version__', 'version_info', 'digit_version']
