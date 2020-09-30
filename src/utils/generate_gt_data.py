import numpy as np
import math
import pybullet as p
import torch
import os, sys
from environment_utils import *

def sample_data(objectID, feature, n_samples=10000):
	"""
	Generates feature data with the ground truth feature function.
	Params:
		objectID -- environment where the feature lives
		feature -- string representing the feature to sample
		n_samples -- number of samples (default: 10000)
	Returns:
		train_points -- configuration space waypoint samples
		regression_labels -- feature labels for train_points
	"""
	n_per_dim = math.ceil(n_samples ** (1 / 7))
	#  we are in 7D radian space
	dim_vector = np.linspace(0, 2 * np.pi, n_per_dim)
	train = []
	labels = []

	for i in range(n_samples):
		sample = np.random.uniform(0, 2 * np.pi, 7)
		if len(sample) < 11:
			sample = np.append(np.append(np.array([0]), sample.reshape(7)), np.array([0,0,0]))
		if feature == "table":
			rl = table_features(objectID, sample)
		elif feature == "human":
			rl = human_features(objectID, sample)
		elif feature == "laptop":
			rl = laptop_features(objectID, sample)
		elif feature == "proxemics":
			rl = proxemics_features(objectID, sample)
		height = table_features(objectID, sample)
		if height >= 0.635:
			train.append(sample)
			labels.append(rl)

	# normalize
	labels = np.array(labels) / max(labels)
	return np.array(train), labels

# -- Distance to Table -- #

def table_features(objectID, waypt):
	"""
	Computes the total feature value over waypoints based on z-axis distance to table.
	---
	Params:
		objectID -- environment where the feature lives
		waypt -- single waypoint
	Returns:
		dist -- scalar feature
	"""
	if len(waypt) < 11:
		waypt = np.append(np.append(np.array([0]), waypt.reshape(7)), np.array([0,0,0]))
	for jointIndex in range(p.getNumJoints(objectID["robot"])):
		p.resetJointState(objectID["robot"], jointIndex, waypt[jointIndex])
	coords = robot_coords(objectID["robot"])
	EEcoord_z = coords[6][2]
	return EEcoord_z

# -- Distance to Laptop -- #

def laptop_features(objectID, waypt):
	"""
	Computes distance from end-effector to laptop in xy coords
	Params:
		objectID -- environment where the feature lives
		waypt -- single waypoint
	Returns:
		dist -- scalar distance where
			0: EE is at more than 0.3 meters away from laptop
			+: EE is closer than 0.3 meters to laptop
	"""
	if len(waypt) < 11:
		waypt = np.append(np.append(np.array([0]), waypt.reshape(7)), np.array([0,0,0]))
	for jointIndex in range(p.getNumJoints(objectID["robot"])):
		p.resetJointState(objectID["robot"], jointIndex, waypt[jointIndex])
	coords = robot_coords(objectID["robot"])
	EE_coord_xy = coords[6][0:2]
	posL, _ = p.getBasePositionAndOrientation(objectID["laptop"])
	laptop_xy = posL[0:2]
	dist = np.linalg.norm(EE_coord_xy - laptop_xy) - 0.4
	if dist > 0:
		return 0
	return -dist

# -- Distance to Human -- #

def human_features(objectID, waypt):
	"""
	Computes distance from end-effector to human in xy coords
	Params:
		objectID -- environment where the feature lives
		waypt -- single waypoint
	Returns:
		dist -- scalar distance where
			0: EE is at more than 0.4 meters away from human
			+: EE is closer than 0.4 meters to human
	"""
	if len(waypt) < 11:
		waypt = np.append(np.append(np.array([0]), waypt.reshape(7)), np.array([0,0,0]))
	for jointIndex in range(p.getNumJoints(objectID["robot"])):
		p.resetJointState(objectID["robot"], jointIndex, waypt[jointIndex])
	coords = robot_coords(objectID["robot"])
	EE_coord_xy = coords[6][0:2]
	posH, _ = p.getBasePositionAndOrientation(objectID["human"])
	human_xy = posH[0:2]
	dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.4
	if dist > 0:
		return 0
	return -dist

# -- Proxemics -- #

def proxemics_features(objectID, waypt):
	"""
	Computes distance from end-effector to human proxemics in xy coords
	Params:
		objectID -- environment where the feature lives
		waypt -- single waypoint
	Returns:
		dist -- scalar distance where
			0: EE is at more than 0.3 meters away from human
			+: EE is closer than 0.3 meters to human
	"""
	if len(waypt) < 11:
		waypt = np.append(np.append(np.array([0]), waypt.reshape(7)), np.array([0,0,0]))
	for jointIndex in range(p.getNumJoints(objectID["robot"])):
		p.resetJointState(objectID["robot"], jointIndex, waypt[jointIndex])
	coords = robot_coords(objectID["robot"])
	EE_coord_xy = coords[6][0:2]
	posH, _ = p.getBasePositionAndOrientation(objectID["human"])
	human_xy = list(posH[0:2])

	# Modify ellipsis distance.
	EE_coord_xy[1] /= 3
	human_xy[1] /= 3
	dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.4
	if dist > 0:
		return 0
	return -dist

def generate_gt_data(feature):
	# create environment instance
	physicsClient = p.connect(p.DIRECT)
	resources_dir = "../../data/resources/"
	p.setAdditionalSearchPath(resources_dir)
	objectID = setup_environment()
	p.setGravity(0, 0, 0)
	p.setRealTimeSimulation(1)
	print("Generating data...")
	# generate training data
	train, labels = sample_data(objectID, feature)
	# Create raw features
	train_raw = np.empty((0, 97), float)
	for dp in train:
		train_raw = np.vstack((train_raw, raw_features(objectID, dp)))
	here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
	np.savez(here+'/data/gtdata/data_{}.npz'.format(feature), x=train_raw, y=labels)
	print("Finished generating data.")

if __name__ == '__main__':
	feat = sys.argv[1]
	generate_gt_data(feat)

