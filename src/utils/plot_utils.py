import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import pybullet as p
from PIL import Image

from utils.environment_utils import *


def angles_to_coords(data, feat, objectID):
	"""
	Transforms an array of 7D Angle coordinates to xyz coordinates except for coffee (to x vector of EE rotation vec.)
	"""
	coords_list = np.empty((0, 3), float)
	for i in range(data.shape[0]):
		waypt = data[i]
		if len(waypt) < 11:
			waypt = np.append(np.append(np.array([0]), waypt.reshape(7)), np.array([0,0,0]))
		for jointIndex in range(p.getNumJoints(objectID["robot"])):
			p.resetJointState(objectID["robot"], jointIndex, waypt[jointIndex])
		if feat == "coffee":
			coords = robot_orientations(objectID["robot"])
			coords_list = np.vstack((coords_list, coords[6][[0,3,6]]))
		else:
			coords = robot_coords(objectID["robot"])
			coords_list = np.vstack((coords_list, coords[6]))
	return coords_list


def plot_gt3D(parent_dir, feat, objectID):
	"""
		Plot the ground truth 3D Half-Sphere for a specific feature.
	"""
	data_file = parent_dir + '/data/gtdata/data_{}.npz'.format(feat)
	npzfile = np.load(data_file)
	train = npzfile['x'][:,:7]
	labels = npzfile['y']
	labels = labels.reshape(len(labels), 1)
	euclidean = angles_to_coords(train, feat, objectID)
	df = pd.DataFrame(np.hstack((euclidean, labels)))
	fig = px.scatter_3d(df, x=0, y=1, z=2, color=3)
	fig.show()


def viz_gt_feature(parent_dir, feat, objectID):
    # get 3D plot traces
    proj_surface, gt_ball = get_3D_plot(parent_dir, feat, objectID, offset = -0.2)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=[
            'Feature Visualization in Space',
            '3D Visualization & Projection'
        ]
    )


    # plot image
    fig.add_trace(px.imshow(Image.open(parent_dir + '/data/images/{}_3D.png'.format(feat))).data[0], row=1, col=1)

    # plot 3D and projection
    fig.add_trace(gt_ball, row=1, col=2)
    fig.add_trace(proj_surface, row=1, col=2)

    fig.update_layout(
        yaxis=dict(showticklabels=False),
        xaxis=dict(showticklabels=False)
    )
    fig.show()
    return

def plot_learned_traj(feature_function, train_data, objectID, feat='table'):
	"""
		Plot the traces labled with the function values of feature_function.
	"""
	output = feature_function(train_data)
	euclidean = angles_to_coords(train_data[:, :7], feat, objectID)
	fig = px.scatter_3d(x=euclidean[:,0], y=euclidean[:,1], z=euclidean[:,2], color=output)
	fig.update_layout(title='Traces with learned function values')
	fig.show()


def plot_learned3D(parent_dir, feature_function, objectID, feat='table', title='Learned function over 3D Reachable Set'):
	"""
		Plot the learned 3D ball over the 10.000 points Test Set in the gt_data
	"""
	data_file = parent_dir + '/data/gtdata/data_{}.npz'.format(feat)
	npzfile = np.load(data_file)
	train = npzfile['x'][:,:7]
	train_raw = np.empty((0, 97), float)
	for waypt in train:
		if len(waypt) < 11:
			waypt = np.append(np.append(np.array([0]), waypt.reshape(7)), np.array([0,0,0]))
		train_raw = np.vstack((train_raw, raw_features(objectID, waypt)))
	labels = feature_function(train_raw)
	euclidean = angles_to_coords(train, feat, objectID)
	fig = px.scatter_3d(x=euclidean[:, 0], y=euclidean[:, 1], z=euclidean[:, 2], color=labels)
	fig.update_layout(title=title)
	fig.show()

def viz_learned_feat(parent_dir, feat, objectID, traces, learned_feature):

    # get learned 3D & projection
    proj_surface, gt_ball = get_3D_plot(parent_dir, feat,
                                        objectID, offset = -0.2, func=learned_feature.function)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            'Traces in 3D Space',
            '3D Visualization & Projection of learned Feature'
        ]
    )

    all_trace_data = np.empty((0, 97), float)
    for idx in range(len(traces)):
        np.flip(traces[idx],axis=0)
        all_trace_data = np.vstack((all_trace_data, traces[idx]))

    # plot traces
    trace_euclidean = angles_to_coords(all_trace_data[:, :7], feat, objectID)
    trace_labels = learned_feature.function(all_trace_data)
    traces=go.Scatter3d(x=trace_euclidean[:,0], y=trace_euclidean[:,1], z=trace_euclidean[:,2], mode='markers',
                                      marker=dict(color=trace_labels.reshape(-1), showscale=False), showlegend=False)
    fig.add_trace(traces, row=1, col=1)

    # plot 3D and projection
    fig.add_trace(gt_ball, row=1, col=2)
    fig.add_trace(proj_surface, row=1, col=2)

    fig.update_layout(
        yaxis=dict(showticklabels=False),
        xaxis=dict(showticklabels=False)
    )
    fig.show()

    return

def get_3D_plot(parent_dir, feat, objectID, offset = -0.2, func=None):
    # load data
    data_file = parent_dir + '/data/gtdata/data_{}.npz'.format(feat)
    npzfile = np.load(data_file)
    train_data = npzfile['x']
    euclidean = angles_to_coords(npzfile['x'][:,:7], feat, objectID)
    # make z axis 0
    euclidean[:,2] = euclidean[:,2] - euclidean[:,2].min()

    if func is None: # do GT labels
        labels = npzfile['y']
    else: # calculate with learned values
        output = func(train_data)
        labels = output.reshape(-1)

    # scale labels to 0-1
    labels = labels - labels.min()
    labels = labels/labels.max()

    # plotting settings
    proj_res = 50

    # plot the GT ball
    gt_ball=go.Scatter3d(x=euclidean[:,0], y=euclidean[:,1], z=euclidean[:,2], mode='markers',
                         marker=dict(color=labels, showscale=True), showlegend=False)

    # create projection of gt
    proj_surface = get_projection_trace(euclidean, labels, feat, offset, proj_res)

    return proj_surface, gt_ball


def get_projection_trace(euclidean, labels, feat, offset, proj_res=50):

    # get scale so that outside is gray
    val_list = [0., 0.01, 0.01, 0.11111111, 0.22222222, 0.33333333, 0.44444444,
       0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.]
    col_list = ['gray', 'gray'] + px.colors.sequential.Plasma
    scale = [[val_list[i], col_list[i]] for i in range(len(col_list))]

    # check which axis to project onto
    if feat in ['laptop', 'proxemics', 'human']: #xy
        idx_1 , idx_2, idx_proj = 0, 1, 2
    else: # yz
        idx_1 , idx_2, idx_proj = 1, 2, 0

    # calculate range of the matrix
    idx_1_min, idx_1_max = euclidean[:,idx_1].min() - 0.1, euclidean[:,idx_1].max() + 0.1
    idx_2_min, idx_2_max = euclidean[:,idx_2].min() - 0.1, euclidean[:,idx_2].max() + 0.1

    # calculate matrix
    proj_matrix = np.zeros((proj_res, proj_res))
    counts = np.zeros((proj_res, proj_res))
    idx_1_space = np.linspace(idx_1_min, idx_1_max, proj_res + 1)
    idx_2_space = np.linspace(idx_2_min, idx_2_max, proj_res + 1)
    for i in range(euclidean.shape[0]):
        if feat in ['laptop', 'proxemics', 'human']: #xy
            temp_idx_2 = np.searchsorted(idx_1_space, euclidean[i, idx_1])
            temp_idx_1 = np.searchsorted(idx_2_space, euclidean[i, idx_2])
            proj_matrix[-(proj_res-temp_idx_1-1)][temp_idx_2] += labels[i]
            counts[-(proj_res-temp_idx_1-1)][temp_idx_2] += 1
        else: # yz
            temp_idx_1 = np.searchsorted(idx_1_space, euclidean[i, idx_1])
            temp_idx_2 = np.searchsorted(idx_2_space, euclidean[i, idx_2])
            proj_matrix[temp_idx_2][proj_res-temp_idx_1-1] += labels[i]
            counts[temp_idx_2][proj_res-temp_idx_1-1] += 1
    proj_matrix /= counts

    # normalize
    proj_matrix = proj_matrix/np.nanmax(proj_matrix)
    proj_matrix[np.isnan(proj_matrix)] = -0.05
#     proj_matrix[~np.isnan(proj_matrix)] = proj_matrix[~np.isnan(proj_matrix)] + 0.08

    # make it plotable as surface
    idx_1_matrix, idx_2_matrix=np.meshgrid(np.linspace(idx_1_min, idx_1_max, proj_res),
                                           np.linspace(idx_2_min, idx_2_max, proj_res))
    proj_cord_offset=(offset + euclidean[:,idx_proj].min())*np.ones(proj_matrix.shape)

    # create the trace
    if feat in ['laptop', 'proxemics', 'human']: #xy
        proj_surface = go.Surface(z=proj_cord_offset,
                              x=idx_1_matrix,
                              y=idx_2_matrix,
                                colorscale= scale,
                              showscale=False,
                              surfacecolor=proj_matrix,
#                               opacity = 0.9
                             )
    else: # yz
        proj_surface = go.Surface(z=idx_2_matrix,
                              x=proj_cord_offset,
                              y=idx_1_matrix,
                                  colorscale= scale,
                              showscale=False,
                              surfacecolor=proj_matrix,
#                                   opacity = 0.9
                             )
    return proj_surface
