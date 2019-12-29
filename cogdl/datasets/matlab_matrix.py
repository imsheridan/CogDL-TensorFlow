import json
import os
import os.path as osp
from itertools import product

import numpy as np
import scipy
import tensorflow as tf
import pickle

from cogdl.data import Data, Dataset, download_url

from . import register_dataset


class MatlabMatrix(Dataset):
    r"""networks from the http://leitang.net/code/social-dimension/data/ or http://snap.stanford.edu/node2vec/

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Blogcatalog"`).
    """

    def __init__(self, root, name, url):
        self.name = name
        self.url = url
        super(MatlabMatrix, self).__init__(root)
        with open(self.processed_paths[0], 'rb') as input:
            self.data = pickle.load(input)

    @property
    def raw_file_names(self):
        splits = [self.name]
        files = ["mat"]
        return ["{}.{}".format(s, f) for s, f in product(splits, files)]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        for name in self.raw_file_names:
            download_url("{}{}".format(self.url, name), self.raw_dir)

    def get(self, idx):
        assert idx == 0
        return self.data

    def process(self):
        path = osp.join(self.raw_dir, "{}.mat".format(self.name))
        smat = scipy.io.loadmat(path)
        adj_matrix, group = smat["network"], smat["group"]

        y = tf.convert_to_tensor(group.todense(), np.float32)

        row_ind, col_ind = adj_matrix.nonzero()
        edge_index = tf.stack([tf.convert_to_tensor(row_ind), tf.convert_to_tensor(col_ind)])
        edge_attr = tf.convert_to_tensor(adj_matrix[row_ind, col_ind])

        data = Data(edge_index=edge_index, edge_attr=edge_attr, x=None, y=y)
        
        with open(self.processed_paths[0], 'wb') as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)


@register_dataset("blogcatalog")
class BlogcatalogDataset(MatlabMatrix):
    def __init__(self):
        dataset, filename = "blogcatalog", "blogcatalog"
        url = "http://leitang.net/code/social-dimension/data/"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(BlogcatalogDataset, self).__init__(path, filename, url)


@register_dataset("flickr")
class FlickrDataset(MatlabMatrix):
    def __init__(self):
        dataset, filename = "flickr", "flickr"
        url = "http://leitang.net/code/social-dimension/data/"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(FlickrDataset, self).__init__(path, filename, url)


@register_dataset("wikipedia")
class WikipediaDataset(MatlabMatrix):
    def __init__(self):
        dataset, filename = "wikipedia", "POS"
        url = "http://snap.stanford.edu/node2vec/"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(WikipediaDataset, self).__init__(path, filename, url)


@register_dataset("ppi")
class PPIDataset(MatlabMatrix):
    def __init__(self):
        dataset, filename = "ppi", "Homo_sapiens"
        url = "http://snap.stanford.edu/node2vec/"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(PPIDataset, self).__init__(path, filename, url)
