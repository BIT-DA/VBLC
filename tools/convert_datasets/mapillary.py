import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from PIL import Image


def convert_to_train_id(file):
    # the label of mapillary is in 0~65 without conversion
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    id_to_train_id = {
        13: 0, 24: 0, 41: 0,
        2: 1, 15: 1,
        17: 2,
        6: 3,
        3: 4,
        45: 5, 47: 5,
        48: 6,
        50: 7,
        30: 8,
        29: 9,
        27: 10,
        19: 11,
        20: 12, 21: 12, 22: 12,
        55: 13,
        61: 14,
        54: 15,
        58: 16,
        57: 17,
        52: 18
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_train_id.items():
        k_mask = label == k
        label_copy[k_mask] = v
        if 'training/' in file:
            n = int(np.sum(k_mask))
            if n > 0:
                sample_class_stats.setdefault(v, 0)
                sample_class_stats[v] += n
    new_file = file.replace('.png', '_labelTrainIds.png')
    assert file != new_file
    if 'training/' in file:
        sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats if 'training/' in file else None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary annotations to TrainIds')
    parser.add_argument('mapillary_path', help='mapillary data path')
    parser.add_argument('--gt-dir', default='v1.2/labels', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    mapillary_path = args.mapillary_path
    out_dir = args.out_dir if args.out_dir else mapillary_path
    mmcv.mkdir_or_exist(out_dir)

    split_names = ['training', 'validation']

    gt_dirs = [osp.join(mapillary_path, split, args.gt_dir) for split in split_names]

    poly_files = []
    for gt_dir in gt_dirs:
        for poly in mmcv.scandir(gt_dir, '.png', recursive=True):
            if not poly.endswith('_labelTrainIds.png'):
                poly_file = osp.join(gt_dir, poly)
                poly_files.append(poly_file)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
