import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandapower as pp

from opf.power import LoadGenerator, NetWrapper, load_case, simplify_net


def generate_samples(manager: NetWrapper, num_samples, load_scale=1.0, delta=0.1):
    p, q = manager.get_load()
    p *= load_scale
    q *= load_scale
    p = LoadGenerator.generate_load_from_random(p, num_samples, delta=delta)
    q = LoadGenerator.generate_load_from_random(q, num_samples, delta=delta)
    load_samples = np.stack((p, q), axis=1)
    return load_samples


def label_sample(manager: NetWrapper, load_sample, powermodels=False):
    """
    Generate the optimal power flow label.
    :param manager: NetworkManger instance.
    :param load_sample: The active and reactive load at each node.
    :return:
    """
    manager.set_load(*load_sample)
    return manager.optimal_ac(powermodels)


def label_samples(manager, load, pool=True, powermodels=False):
    """
    Labels samples using :method:`label_sample`.
    :param manager: NetworkManager instance.
    :param load: Active and reactive loads on each node per sample.
    :param pool: If true then use a process pool.
    :return: A list of tuples (res_bus, res_gen, res_ext_grid) or None if OPF did not converge..
    """
    f = partial(label_sample, manager, powermodels=powermodels)
    if pool:
        with Pool() as p:
            labels = list(p.map(f, load))
    else:
        labels = list(map(f, load))
    load, labels = zip(*filter(lambda x: x[1] is not None, zip(load, labels)))
    bus, gen, ext = zip(*labels)
    return load, bus, gen, ext


def main():
    parser = argparse.ArgumentParser(description="Generate OPF dataset.")
    parser.add_argument(
        "case",
        type=str,
        help="Test case to use. Either specify a pandapower casename or a test case file ending with *.m",
    )
    parser.add_argument(
        "train_samples", type=int, help="Number of unlabeled samples to generate."
    )
    parser.add_argument(
        "test_samples", type=int, help="Number of labeled samples to generate."
    )
    parser.add_argument(
        "-d",
        "--data",
        metavar="-d",
        type=str,
        help="The data directory.",
        default="./data",
    )
    parser.add_argument("--scale", default=1.0, type=float, help="Scale the load.")
    parser.add_argument("--name", default=None, type=str, help="Output filename.")
    parser.add_argument(
        "--powermodels",
        default=False,
        action="store_true",
        help="If set then will use PowerModels instead of PYPOWER solver.",
    )
    parser.add_argument(
        "--pool",
        default=False,
        action="store_true",
        help="If set then will use a multiprocessing pool to accelerate ACOPF.",
    )
    args = parser.parse_args()

    # load matpower case
    case = load_case(args.case)
    # simplify case so that there is one generator per bus
    case = simplify_net(case)
    manager = NetWrapper(case, per_unit=True)
    train_load = generate_samples(manager, args.train_samples, load_scale=args.scale)

    test_load, test_bus, test_gen, test_ext = None, None, None, None
    if args.test_samples > 0:
        test_load = generate_samples(manager, args.test_samples, load_scale=args.scale)
        test_load, test_bus, test_gen, test_ext = label_samples(
            manager, test_load, pool=args.pool, powermodels=args.powermodels
        )

    if args.name is None:
        # get the filename without extension
        args.name = os.path.basename(os.path.splitext(args.case)[0])

    # save pandapower test case
    pp.to_pickle(case, os.path.join(args.data, f"{args.name}.pkl"))
    # save numpy dataset
    filename = os.path.join(args.data, args.name, "train.npz")
    np.savez(
        filename,
        train_load=train_load,
        test_load=test_load,
        test_bus=test_bus,
        test_gen=test_gen,
        test_ext=test_ext,
    )
    print(
        f"Generation complete!"
        f"Train samples: {len(train_load)}"
        f"Test samples: {len(test_load)}"
    )


if __name__ == "__main__":
    main()
