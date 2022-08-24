# Modifications: - keep_soft in format_results

import os.path as osp

import matplotlib
import mmcv
import numpy as np
from PIL import Image
from matplotlib import cm
matplotlib.use('Agg')

from .builder import DATASETS
from .cityscapes import CityscapesDataset
from .custom import CustomDataset


@DATASETS.register_module()
class ACDCDataset(CustomDataset):
    """ACDCDataset dataset."""

    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        super(ACDCDataset, self).__init__(
            img_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)

    def results2img(self, results, imgfile_prefix, do_palette=True, indices=None, keep_soft=False):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            do_palette (bool): whether convert output to color label for
                visualization.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.
            keep_soft (bool): whether keep soft prediction. Default: False

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            if keep_soft:
                if do_palette:
                    assert len(result.shape) == 2, f'visualize result with more than one channel not supported'
                    png_filename = osp.join(imgfile_prefix, f'{basename}.png')
                    output = Image.fromarray(np.uint8(cm.get_cmap('inferno')(result)*255))
                    output.save(png_filename)
                    # result_files.append(png_filename)  # fix progress bar
                pkl_filename = osp.join(imgfile_prefix, f'{basename}.pkl')
                mmcv.dump(result, pkl_filename)
                result_files.append(pkl_filename)
            else:
                png_filename = osp.join(imgfile_prefix, f'{basename}.png')

                if do_palette:
                    result = CityscapesDataset.city_convert_to_label_id(result)
                    output = Image.fromarray(result.astype(np.uint8)).convert('P')
                    import cityscapesscripts.helpers.labels as CSLabels
                    palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
                    for label_id, label in CSLabels.id2label.items():
                        palette[label_id] = label.color

                    output.putpalette(palette)
                else:
                    output = Image.fromarray(result.astype(np.uint8))
                output.save(png_filename)
                result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       do_palette=True,
                       indices=None,
                       keep_soft=False):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            do_palette (bool): whether convert output to color label for
                visualization. Default: True
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.
            keep_soft (bool): whether keep soft prediction. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'
        # assert (not do_palette) or (not keep_soft), 'no palette scheme for soft predictions.'

        result_files = self.results2img(results, imgfile_prefix, do_palette,
                                        indices, keep_soft)

        return result_files
