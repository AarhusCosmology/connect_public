###############################################################################
# Log+Linear scale used to plot the CMB power spectrum by the ESA Planck team #
###############################################################################

# Note: import the whole module, not just the class,
#       since the scale must be registered (see last line)

import numpy as np
from numpy import ma
MaskedArray = ma.MaskedArray
from math import floor
from matplotlib import scale as mscale
from matplotlib.transforms import Transform
from matplotlib.ticker import Formatter, FixedLocator

nonpos = "mask"
change = 200.0
factor = 500.

def new_change(new_c):
    global change
    change = new_c

def _mask_nonpos(a):
    """
    Return a Numpy masked array where all non-positive 1 are
    masked.  If there are no non-positive, the original array
    is returned.
    """
    mask = a <= 0.0
    if mask.any():
        return ma.MaskedArray(a, mask=mask)
    return a

def _clip_smaller_than_one(a):
    a[a <= 0.0] = 1e-300
    return a

class PlanckScale(mscale.ScaleBase):
    """
    Scale used by the Planck collaboration to plot Temperature power spectra:
    base-10 logarithmic up to l=50, and linear from there on.
    Care is taken so non-positive values are not plotted.
    """
    name = 'planck'

    def __init__(self, axis, **kwargs):
        pass

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(
            FixedLocator(
                np.concatenate((np.array([2, 10, change]),
                                np.arange(500, 2500, 500)))))
        axis.set_minor_locator(
            FixedLocator(
                np.concatenate((np.arange(2, 10),
                                np.arange(10, 50, 10),
                                np.arange(floor(change/100), 2500, 100)))))

    def get_transform(self):
        """
        Return a :class:`~matplotlib.transforms.Transform` instance
        appropriate for the given logarithm base.
        """
        return self.PlanckTransform(nonpos)
        
    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to positive values.
        """
        return (vmin <= 0.0 and minpos or vmin,
                vmax <= 0.0 and minpos or vmax)

    class PlanckTransform(Transform):
        input_dims   = 1
        output_dims  = 1
        is_separable = True
        has_inverse = True
        
        def __init__(self, nonpos):
            Transform.__init__(self)
            if nonpos == 'mask':
                self._handle_nonpos = _mask_nonpos
            else:
                self._handle_nonpos = _clip_nonpos

        def transform(self, values):
            values = np.asanyarray(values)
            ndim = values.ndim
            values = values.reshape((-1, self.input_dims))
            res = self.transform_affine(self.transform_non_affine(values))
            if ndim == 0:
                assert not np.ma.is_masked(res)
                try:
                    try:
                        return res[0, 0]
                    except:
                        return res[0]
                except:
                    return res
            if ndim == 1:
                return res.reshape(-1)
            elif ndim == 2:
                return res
            raise ValueError(
                "Input values must have shape (N x {dims}) "
                "or ({dims}).".format(dims=self.input_dims))

        def transform_non_affine(self, a):
            lower   = a[np.where(a<=change)]
            greater = a[np.where(a> change)]
            if lower.size:
                lower = self._handle_nonpos(lower * 10.0)/10.0
                if isinstance(lower, MaskedArray):
                    lower = ma.log10(lower)
                else:
                    lower = np.log10(lower)
                lower = factor*lower
            if greater.size:
                greater = (factor*np.log10(change) + (greater-change))
            # Only low
            if not(greater.size):
                return lower
            # Only high
            if not(lower.size):
                return greater
            return np.concatenate((lower, greater))
        def inverted(self):
            return PlanckScale.InvertedPlanckTransform()

    class InvertedPlanckTransform(Transform):
        input_dims   = 1
        output_dims  = 1
        is_separable = True
        has_inverse = True

        def transform(self, values):
            values = np.asanyarray(values)
            ndim = values.ndim
            values = values.reshape((-1, self.input_dims))
            res = self.transform_affine(self.transform_non_affine(values))
            if ndim == 0:
                assert not np.ma.is_masked(res)
                try:
                    return res[0, 0]
                except:
                    return res[0]
            if ndim == 1:
                return res.reshape(-1)
            elif ndim == 2:
                return res
            raise ValueError(
                "Input values must have shape (N x {dims}) "
                "or ({dims}).".format(dims=self.input_dims))
        
        def transform_non_affine(self, a):
            lower   = a[np.where(a<=factor*np.log10(change))]
            greater = a[np.where(a> factor*np.log10(change))]
            if lower.size:
                if isinstance(lower, MaskedArray):
                    lower = ma.power(10.0, lower/float(factor))
                else:
                    lower = np.power(10.0, lower/float(factor))
            if greater.size:
                greater = (greater + change - factor*np.log10(change))
            # Only low
            if not(greater.size):
                return lower
            # Only high
            if not(lower.size):
                return greater
            return np.concatenate((lower, greater))
        def inverted(self):
            return PlanckScale.PlanckTransform(nonpos)

# Finished. Register the scale!
mscale.register_scale(PlanckScale)
