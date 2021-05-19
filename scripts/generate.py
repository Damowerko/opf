#!/usr/bin/python
import argparse

from pyopf.data import generate_samples, label_samples, save_data
from pyopf.power import NetworkManager, load_case

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OPF dataset.")
    parser.add_argument("case", type=str, help="Test case to use.")
    parser.add_argument("train_samples", type=int, help="Number of unlabeled samples to generate.")
    parser.add_argument("test_samples", type=int, help="Number of labeled samples to generate.")
    parser.add_argument("-d", "--data", metavar="-d", type=str, help="The data directory.", default="./data")
    parser.add_argument("--scale", default=1.0, type=float, help="Scale the load.")
    parser.add_argument("--name", default=None, type=str, help="The filename.")
    args = parser.parse_args()

    case = load_case(args.case, args.data)
    manager = NetworkManager(case)
    train_samples = generate_samples(manager, args.train_samples, load_scale=args.scale)

    test_samples = None
    test_labels = None
    if args.test_samples > 0:
        test_samples = generate_samples(manager, args.test_samples, load_scale=args.scale)
        test_samples, test_labels = label_samples(manager, test_samples, True)

    if args.name is None:
        args.name = args.case

    save_data(args.name, train_samples, test_samples, test_labels, args.data)
    print(f"Generation complete!"
          f"Train samples: {len(train_samples)}"
          f"Test samples: {len(test_samples)}")
