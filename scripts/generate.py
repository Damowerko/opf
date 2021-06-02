#!/usr/bin/python
import argparse
import os

import numpy as np

from opf.generate import generate_samples, label_samples
from opf.power import load_case, NetWrapper

def main():
    parser = argparse.ArgumentParser(description="Generate OPF dataset.")
    parser.add_argument("case", type=str, help="Test case to use.")
    parser.add_argument("train_samples", type=int, help="Number of unlabeled samples to generate.")
    parser.add_argument("test_samples", type=int, help="Number of labeled samples to generate.")
    parser.add_argument("-d", "--data", metavar="-d", type=str, help="The data directory.", default="./data")
    parser.add_argument("--scale", default=1.0, type=float, help="Scale the load.")
    parser.add_argument("--name", default=None, type=str, help="The filename.")
    args = parser.parse_args()

    case = load_case(args.case, args.data)
    manager = NetWrapper(case, per_unit=True)
    print(f"Generating {args.train_samples} train samples...")
    train_load = generate_samples(manager, args.train_samples, load_scale=args.scale)

    test_load, test_bus, test_gen, test_ext = None, None, None, None
    if args.test_samples > 0:
        print(f"Generating {args.test_samples} test samples...")
        test_load = generate_samples(manager, args.test_samples, load_scale=args.scale)
        print(f"Labeling test sampels...")
        test_load, test_bus, test_gen, test_ext = label_samples(manager, test_load, False)

    if args.name is None:
        args.name = args.case

    filename = os.path.join(args.data, args.case + ".npz")
    np.savez(filename,
             train_load=train_load,
             test_load=test_load,
             test_bus=test_bus,
             test_gen=test_gen,
             test_ext=test_ext)
    print(f"Generation complete!"
          f"Train samples: {len(train_load)}"
          f"Test samples: {len(test_load)}")

if __name__ == "__main__":
    main()