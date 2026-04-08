import argparse
import os

import numpy as np
import torch

from .utils import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR


def visualize(checkpoint_path, config_path, output_path='structure_vq.xyz'):
    from .gnn_morse_fitter import load_config
    from .data import DFTDataset
    from .models import GNNMorseModel
    from .plotting import write_structure_xyz

    config = load_config(config_path)

    dataset = DFTDataset(
        config['datasets'],
        knn_config=config['knn_edges'],
        first_n_frames=1,
    )
    dataset.build_graphs()
    graph = dataset.graphs[0]

    gnn_raw = config['gnn']
    if 'vq_max_codes' in gnn_raw:
        n_subtypes = gnn_raw['vq_max_codes']

    gnn_config = gnn_raw.copy()
    gnn_config['core_elements'] = config.get('core_elements', dataset.elements)
    gnn_config['vq_max_codes'] = n_subtypes

    model = GNNMorseModel(
        num_pairs=len(dataset.pair_names),
        num_elements=len(dataset.elements),
        elements=dataset.elements,
        pair_names=dataset.pair_names,
        gnn_config=gnn_config,
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.vq_active = True

    # Ensure output path ends in .xyz
    if not output_path.endswith('.xyz'):
        output_path = os.path.splitext(output_path)[0] + '.xyz'

    write_structure_xyz(model, dataset, 'cpu', config, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate VQ atom type XYZ for OVITO')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--config', required=True,
                        help='Path to config .yaml file')
    parser.add_argument('--output', default='structure_vq.xyz',
                        help='Output XYZ path (default: structure_vq.xyz)')
    args = parser.parse_args()
    visualize(args.checkpoint, args.config, args.output)


if __name__ == '__main__':
    main()
