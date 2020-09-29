import pybullet as p
import time
import argparse
import pickle
import numpy as np
import torch

from utils.learned_feature import LearnedFeature
from utils.environment_utils import *
from utils.plot_utils import *


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/user_data",
        help="Path to dir where features should be loaded.",
    )

    parser.add_argument(
        "--resources-dir",
        type=str,
        default="../data/resources",
        help="Path to dir where environment resources are stored.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse experimental arguments.
    args = parse_arguments()

    # Connect to physics simulator.
    physicsClient = p.connect(p.DIRECT) #or p.DIRECT for non-graphical version

    # Add path to data resources for the environment.
    p.setAdditionalSearchPath(args.resources_dir)

    # Setup the environment.
    objectID = setup_environment()

    # Get rid of gravity and make simulation happen in real time.
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(1)

    # Visualize features.
    features = ["table", "laptop", "proxemics"]
    for feature in features:
        print("Visualizing {} feature.".format(feature))
        traces = pickle.load(open("{}/{}_1.p".format(args.data_dir, feature), "rb" ) )
        model = torch.load("{}/{}_1.pt".format(args.data_dir, feature))
        viz_gt_feature("../", feature, objectID)
        print("Press ENTER when you're done visualizing the ground truth feature.")
        input()
        viz_learned_feat("../", feature, objectID, traces, model)
        print("Press ENTER when you're done visualizing the learned feature.")
        input()
        print("Fill out the survey for {} and press ENTER when done.".format(feature))
        input()

    # Disconnect once the session is over.
    p.disconnect()
