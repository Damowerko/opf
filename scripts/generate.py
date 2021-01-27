#!/usr/bin/python
import argparse
from pyopf.data import generate_samples, label_samples, save_data
from pyopf.power import NetworkManager, load_case
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OPF dataset.")
    parser.add_argument("case", type=str, help="Test case to use.")
    parser.add_argument("train_samples", type=int, help="Number of unlabeled samples to generate.")
    parser.add_argument("test_samples", type=int, help="Number of labeled samples to generate.")
    parser.add_argument("-d", "--data", metavar="-d", type=str, help="The data directory.", default="./data")
    parser.add_argument("--scale", default=1.0, type=float, help="Scale the load.")
    args = parser.parse_args()

    case = load_case(args.case, args.data)
    manager = NetworkManager(case)
    train_samples = generate_samples(manager, args.train_samples, load_scale=args.scale)

    test_samples = None
    test_labels = None
    if args.test_samples > 0:
        test_samples = generate_samples(manager, args.test_samples, load_scale=args.scale)
        test_samples, test_labels = label_samples(manager, test_samples, True)
        samples = np.stack(test_samples)
        labels = np.stack(test_labels)

    save_data(args.case, train_samples, test_samples, test_labels, args.data)
