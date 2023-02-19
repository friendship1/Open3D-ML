import hashlib
from typing import Callable
import numpy as np

from os import makedirs, listdir
from os.path import exists, join, splitext


def make_dir(folder_name):
    """Create a directory.

    If already exists, do nothing
    """
    if not exists(folder_name):
        makedirs(folder_name)


def get_hash(x: str):
    """Generate a hash from a string."""
    h = hashlib.md5(x.encode())
    return h.hexdigest()

hash_remap = {
    '21cae1385e5df4f7eda6f6f8ca7cc686':'19aaf3d94dcb2f5641af953308436e16', # RandLaNetNoXY -> RandLaNet
    '386c10d76bd9abe46376ce3714937afd':'19aaf3d94dcb2f5641af953308436e16', # RandLaNetNoXYBase -> RandLaNet
    'a26e0d3e7d15b654eb94accff07789ae':'19aaf3d94dcb2f5641af953308436e16', # RandLaNetNoXYBase2 -> RandLaNet
    '09184da840ea6a569f1e66aee5928c98':'19aaf3d94dcb2f5641af953308436e16', # RandLaNetNoXYZ -> RandLaNet
    '678f491374bfac84b31fb71c956750d2':'19aaf3d94dcb2f5641af953308436e16', # RandLaNetBN -> RandLaNet
}

class Cache(object):
    """Cache converter for preprocessed data."""

    def __init__(self, func: Callable, cache_dir: str, cache_key: str):
        """Initialize.

        Args:
            func: preprocess function of a model.
            cache_dir: directory to store the cache.
            cache_key: key of this cache
        Returns:
            class: The corresponding class.
        """
        self.func = func
        self.cache_dir = join(cache_dir, hash_remap.get(cache_key, cache_key))
        make_dir(self.cache_dir)
        self.cached_ids = [splitext(p)[0] for p in listdir(self.cache_dir)]

    def __call__(self, unique_id: str, *data):
        """Call the converter. If the cache exists, load and return the cache,
        otherwise run the preprocess function and store the cache.

        Args:
            unique_id: A unique key of this data.
            data: Input to the preprocess function.

        Returns:
            class: Preprocessed (cache) data.
        """
        fpath = join(self.cache_dir, str('{}.npy'.format(unique_id)))

        if not exists(fpath):
            output = self.func(*data)

            self._write(output, fpath)
            self.cached_ids.append(unique_id)
        else:
            output = self._read(fpath)

        return self._read(fpath)

    def _write(self, x, fpath):
        np.save(fpath, x)

    def _read(self, fpath):
        return np.load(fpath, allow_pickle=True).item()
