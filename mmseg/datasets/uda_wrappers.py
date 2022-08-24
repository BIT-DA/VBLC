from .builder import DATASETS
from . import CityscapesDataset

import json
import os.path as osp

import mmcv
from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log
from mmseg.core import add_prefix

import torch
import numpy as np


def get_rcs_class_probs(data_root, temperature, json_file='sample_class_stats.json'):
    with open(osp.join(data_root, json_file), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class UDADataset:
    """A wrapper of unsupervised domain adaptation dataset.

    Args:
        source (:obj:`Dataset`): source dataset.
        target (:obj:`Dataset`): target dataset.
        cfg (dict): configuration.
    """

    def __init__(self, source, target, cfg):
        self.source = source
        self.target = target
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE

        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE

        # rare-class-sampling
        rcs_cfg = cfg.get('rare_class_sampling')
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']
            self.rcs_class_num = len(self.CLASSES)
            scs_json_file = 'sample_class_stats.json'
            swc_json_file = 'samples_with_class.json'
            if self.rcs_class_num != 19:
                scs_json_file = f'sample_class_stats_{self.rcs_class_num}.json'
                swc_json_file = f'samples_with_class_{self.rcs_class_num}.json'

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                cfg['source']['data_root'], self.rcs_class_temp, scs_json_file)
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(
                    osp.join(cfg['source']['data_root'],
                             swc_json_file), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                if isinstance(self.source, CityscapesDataset):
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                s1 = self.source[i1]
        i2 = np.random.choice(range(len(self.target)))
        s2 = self.target[i2]

        return s1, s2

    def __getitem__(self, idx):
        """Get data after pipeline for each source/target dataset.

        Args:
            idx (int): Index of data.

        Returns:
            dict (Dict[List[List[data]]]): Training/test data (with annotation if `test_mode` is set
                False).
                Note that the dict could be multi-level, with the following structure:
                dict(keys include `img`, `img_metas`, `gt_semantic_seg`)
                    list(source/target split)
                        list(different datasets split)
                            actual data(converted when loaded)
        """
        res_tree = dict(img=[[], []], img_metas=[[], []], gt_semantic_seg=[[], []])

        if self.rcs_enabled:
            s, t = self.get_rare_class_sample()
        else:
            s = self.source[idx // len(self.target)]
            t = self.target[idx % len(self.target)]

        res_tree['img'][0].append(s['img'])
        s['img_metas'].data['dataset'] = type(self.source).__name__
        res_tree['img_metas'][0].append(s['img_metas'])
        res_tree['gt_semantic_seg'][0].append(s['gt_semantic_seg'])

        res_tree['img'][1].append(t['img'])
        t['img_metas'].data['dataset'] = type(self.target).__name__
        res_tree['img_metas'][1].append(t['img_metas'])
        res_tree['gt_semantic_seg'][1].append(t['gt_semantic_seg'])

        return res_tree

    def __len__(self):
        return len(self.source) * len(self.target)


@DATASETS.register_module()
class UDAMultiDataset:
    """A wrapper of unsupervised domain adaptation dataset.
    A substitute for UDACompleteDataset, __len__ is given by
    multiplication of max(len(sources)) and max(len(targets)),
    which means all sampling conditions are not present.

    Args:
        source_datasets (list[:obj:`Dataset`]): A list of source datasets.
        target_datasets (list[:obj:`Dataset`]): A list of target datasets.
        cfg (dict): configuration.
    """

    def __init__(self, source_datasets, target_datasets, cfg):
        self.sources = source_datasets
        self.targets = target_datasets
        # self.ignore_index = self.sources[0].ignore_index
        self.CLASSES = self.targets[0].CLASSES
        self.PALETTE = self.targets[0].PALETTE
        len_src = 1
        len_tgt = 1
        for dataset in self.sources:
            # assert dataset.ignore_index == self.ignore_index
            assert dataset.CLASSES == self.CLASSES
            assert dataset.PALETTE == self.PALETTE
            len_src = max(len_src, len(dataset))
        for dataset in self.targets:
            # assert dataset.ignore_index == self.ignore_index
            assert dataset.CLASSES == self.CLASSES
            assert dataset.PALETTE == self.PALETTE
            len_tgt = max(len_tgt, len(dataset))
        self._len = len_src * len_tgt

    def __getitem__(self, idx):
        """Get data after pipeline for each source/target dataset.

        Args:
            idx (int): Index of data.

        Returns:
            dict (Dict[List[List[data]]]): Training/test data (with annotation if `test_mode` is set
                False).
                Note that the dict could be multi-level, with the following structure:
                dict(keys include `img`, `img_metas`, `gt_semantic_seg`)
                    list(source/target split)
                        list(different datasets split)
                            actual data(converted when loaded)
        """
        res_tree = dict(img=[[], []], img_metas=[[], []], gt_semantic_seg=[[], []])
        for source in self.sources:
            s = source[idx % len(source)]
            res_tree['img'][0].append(s['img'])
            s['img_metas'].data['dataset'] = type(source).__name__
            res_tree['img_metas'][0].append(s['img_metas'])
            res_tree['gt_semantic_seg'][0].append(s['gt_semantic_seg'])
        for target in self.targets:
            t = target[idx % len(target)]
            res_tree['img'][1].append(t['img'])
            t['img_metas'].data['dataset'] = type(target).__name__
            res_tree['img_metas'][1].append(t['img_metas'])
            res_tree['gt_semantic_seg'][1].append(t['gt_semantic_seg'])

        return res_tree

    def __len__(self):
        """Returns length of composited datasets"""
        return self._len


@DATASETS.register_module()
class MultiTestDataset:
    """A wrapper of multiple dataset for individual test.

    Args:
        datasets (list[:obj:`Dataset`]): A list of target datasets.
    """

    def __init__(self, datasets, cfg):
        self.datasets = datasets
        self.num_ds = len(datasets)
        self.ignore_index = self.datasets[0].ignore_index
        self.CLASSES = self.datasets[0].CLASSES
        self.PALETTE = self.datasets[0].PALETTE
        self._len = 1
        for dataset in self.datasets:
            assert dataset.ignore_index == self.ignore_index
            assert dataset.CLASSES == self.CLASSES
            assert dataset.PALETTE == self.PALETTE
            self._len = max(self._len, len(dataset))
        self.default_img = [None for _ in range(self.num_ds)]

    def __getitem__(self, idx):
        """Get data after pipeline for each dataset.

        Args:
            idx (int): Index of data.

        Returns:
            dict (Dict[List[data]]): Test data (with annotation if `test_mode` is set
                False).
        """
        res_tree = dict(img=[], img_metas=[])
        for i, dataset in enumerate(self.datasets):
            if idx < len(dataset):
                d = dataset[idx]  # 'img_metas' is in type List[DataContainer], and 'img' is in type List[tensor]
                res_tree['img'].append(d['img'])
                d['img_metas'][0].data['dataset'] = type(dataset).__name__
                res_tree['img_metas'].append(d['img_metas'])
                if self.default_img[i] is None:
                    self.default_img[i] = torch.zeros_like(d['img'][0])
            else:
                res_tree['img'].append([self.default_img[i].clone()])
                res_tree['img_metas'].append([DC({'filename': None, 'dataset': type(dataset).__name__}, cpu_only=True)])

        return res_tree

    def __len__(self):
        return self._len

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[numpy.ndarray]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]

        neat_results = [[] for _ in range(self.num_ds)]
        for result in results:
            for i in range(self.num_ds):
                neat_results[i].extend(result[i])

        eval_results = {}
        key_results = {}
        for i in range(self.num_ds):
            dataset = self.datasets[i]
            ds_name = type(dataset).__name__.replace('Dataset', '')
            print_log(f'{ds_name} Results', logger)
            ds_result = dataset.evaluate(neat_results[i], metric, logger, gt_seg_maps, **kwargs)
            for m in metric:
                if m not in key_results:
                    key_results[m] = []
                key_results[m].append(ds_result[m])
            eval_results.update(add_prefix(ds_result, ds_name))
        print_log('eval complete', logger)

        for m in metric:
            eval_results.update({m: np.mean(key_results[m])})

        return eval_results
