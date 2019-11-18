import random
import numpy as np

from albumentations import DualTransform
from skimage.transform import PiecewiseAffineTransform, warp


class CustomPiecewiseAffineTransform(DualTransform):
    """
    Add sine-wave piecewise affine transform to the image

    Args:
        phase_shift_limit, amplitude_limit, w_limit: parameters of a sine wave
        value: padding value
        p: probability of applying the transform. Default: 0.5.
    """

    def __init__(
            self,
            phase_shift_limit=(0, 50),
            amplitude_limit=(3, 5),
            w_limit=(2, 4),
            value=255,
            always_apply=False,
            p=0.2,
    ):
        super(CustomPiecewiseAffineTransform, self).__init__(always_apply, p)

        self._tform = PiecewiseAffineTransform()

        self.phase_shift_limit = phase_shift_limit
        self.amplitude_limit = amplitude_limit
        self.w_limit = w_limit
        self.value = value

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError(
            "Method get_params_dependent_on_targets is not implemented in class " + self.__class__.__name__)

    def get_transform_init_args_names(self):
        raise NotImplementedError(
            "Method get_transform_init_args_names is not implemented in class " + self.__class__.__name__)

    def _piecewise_affine_transform(self, img, phase_shift, amplitude, w):
        rows, cols = img.shape[0], img.shape[1]

        src_cols = np.linspace(0, cols, 20)
        src_rows = np.linspace(0, rows, 10)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]

        # add sinusoidal oscillation to row coordinates
        dst_rows = src[:, 1] + np.cos(np.linspace(0, w * np.pi, src.shape[0]) + phase_shift) * amplitude
        dst_cols = src[:, 0]
        dst_rows *= 1.2
        dst = np.vstack([dst_cols, dst_rows]).T

        self._tform.estimate(src, dst)

        out_rows = rows
        out_cols = cols
        out = warp(img, self._tform, output_shape=(out_rows, out_cols))

        return out

    def apply(self, img, **params):
        phase_shift = random.uniform(*self.phase_shift_limit)
        amplitude = random.uniform(*self.amplitude_limit)
        w = random.uniform(*self.w_limit)

        return self._piecewise_affine_transform(img, phase_shift, amplitude, w)
