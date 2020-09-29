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
        "--feature",
        type=str,
        required=True,
        help="Feature to be taught",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="../data/user_data/",
        help="Path to dir where traces should be saved",
    )

    parser.add_argument(
        "--resources-dir",
        type=str,
        default="../data/resources",
        help="Path to dir where environment resources are stored.",
    )

    args = parser.parse_args()
    return args


def topview1():
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-89.9, cameraPitch=-89.9,
                                 cameraTargetPosition=[0.0,0.0,0.0])

def topview2():
    p.resetDebugVisualizerCamera(cameraDistance=2.1, cameraYaw=-173.43, cameraPitch=-72.72,
                                 cameraTargetPosition=[-0.2,0,0.2])

def sideview():
    view = np.random.choice(np.arange(4), 1)
    if view == 0:
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-270,
                                 cameraPitch=-15,
                                 cameraTargetPosition=[0,0,0.5])
    elif view == 1:
        p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=-340,
                                 cameraPitch=-15,
                                 cameraTargetPosition=[0,0,0.5])
    elif view == 2:
        p.resetDebugVisualizerCamera(cameraDistance=1.7, cameraYaw=-225,
                                 cameraPitch=-7,
                                 cameraTargetPosition=[0,0,0.5])
    elif view == 3:
        p.resetDebugVisualizerCamera(cameraDistance=1.7, cameraYaw=-385,
                                 cameraPitch=-15,
                                 cameraTargetPosition=[0,0,0.5])


def replay_trace(trace, objectID):
    for waypt in trace:
        for jointIndex in range(p.getNumJoints(objectID["robot"])):
            p.resetJointState(objectID["robot"], jointIndex, waypt[jointIndex])
        time.sleep(0.01)

def next_trace(feature, objectID):
    move_robot(objectID["robot"])
    if feature in ["laptop"]:
        topview1()
    elif feature in ["table"]:
        sideview()
    elif feature in ["human", "proxemics"]:
        topview2()

def envsetup(args, direct=False):
    # Connect to physics simulator.
    if direct:
        physicsClient = p.connect(p.DIRECT)
    else:
        physicsClient = p.connect(p.GUI)

    # Add path to data resources for the environment.
    p.setAdditionalSearchPath(args.resources_dir)

    # Setup the environment.
    objectID = setup_environment()

    # Get rid of gravity and make simulation happen in real time.
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(1)

    return objectID


if __name__ == "__main__":
    # Parse experimental arguments.
    args = parse_arguments()

    # Visualize ground truth feature.
    objectID = envsetup(args, direct=True)
    viz_gt_feature("../", args.feature, objectID)
    p.disconnect()

    print("Press ENTER when done visualizing.")
    input()

    # Start GUI
    objectID = envsetup(args)

    # Collect data.
    recordButton = p.addUserDebugParameter("Start Recording", 1, 0, 0)
    stopButton = p.addUserDebugParameter("Stop Recording", 1, 0, 0)
    replayButton = p.addUserDebugParameter("Replay Trace", 1, 0, 0)
    saveButton = p.addUserDebugParameter("Save Trace", 1, 0, 0)
    nextButton = p.addUserDebugParameter("Next Trace", 1, 0, 0)
    labelButton = p.addUserDebugParameter("Give Start-End Labels", 1, 0, 0)
    top1Button = p.addUserDebugParameter("Top-down View #1", 1, 0, 0)
    top2Button = p.addUserDebugParameter("Top-down View #2", 1, 0, 0)
    sideButton = p.addUserDebugParameter("Side View", 1, 0, 0)
    recordNum = 0
    stopNum = 0
    replayNum = 0
    saveNum = 0
    nextNum = 0
    labelNum = 0
    top1Num = 0
    top2Num = 0
    sideNum = 0

    if args.feature in ["human"]:
        N_QUERIES = 5
    else:
        N_QUERIES = 10

    trace = []
    traces = []
    start_labels = [1.0] * N_QUERIES
    end_labels = [0.0] * N_QUERIES
    queries = 0
    record = False
    print("Attempting trace #{}. Please place the robot's end-effector in a highly-expressed feature region, and when ready press Start Recording.".format(queries+1))

    if args.feature in ["laptop"]:
        topview1()
    elif args.feature in ["table"]:
        sideview()
    elif args.feature in ["human", "proxemics"]:
        topview2()

    while(queries < N_QUERIES):
        recordPushes = p.readUserDebugParameter(recordButton)
        stopPushes = p.readUserDebugParameter(stopButton)
        replayPushes = p.readUserDebugParameter(replayButton)
        savePushes = p.readUserDebugParameter(saveButton)
        nextPushes = p.readUserDebugParameter(nextButton)
        labelPushes = p.readUserDebugParameter(labelButton)
        top1Pushes = p.readUserDebugParameter(top1Button)
        top2Pushes = p.readUserDebugParameter(top2Button)
        sidePushes = p.readUserDebugParameter(sideButton)

        if top1Pushes > top1Num:
            print("Changing the view to top-down #1.")
            top1Num = top1Pushes
            topview1()

        if top2Pushes > top2Num:
            print("Changing the view to top-down #2.")
            top2Num = top2Pushes
            topview2()

        if sidePushes > sideNum:
            print("Changing the view to side.")
            sideNum = sidePushes
            sideview()

        if labelPushes > labelNum:
            labelNum = labelPushes
            if queries > 0:
                print("What would you like to label the trace start? Enter a number from 0-10, where 10 is highly-expressed and 0 is lowly-expressed.")
                line = input()
                if line in [str(i) for i in range(11)]:
                    start_labels[queries-1] = int(line) / 10.0
                print("What would you like to label the trace end? Enter a number from 0-10, where 10 is highly-expressed and 0 is lowly-expressed.")
                line = input()
                if line in [str(i) for i in range(11)]:
                    end_labels[queries-1] = int(line) / 10.0
            else:
                print("No saved traces to label yet.")

        if nextPushes > nextNum:
            print("Attempting trace #{}. Please place the robot's end-effector in a highly-expressed feature region, and when ready press Start Recording.".format(queries+1))
            nextNum = nextPushes
            trace = []
            next_trace(args.feature, objectID)

        if savePushes > saveNum:
            saveNum = savePushes
            if record == False:
                if len(trace) > 0:
                    raw_trace = []
                    for waypt in trace:
                        raw_trace.append(raw_features(objectID, waypt))

                    # Save trace.
                    raw_trace.reverse()
                    traces.append(np.array(raw_trace))
                    print("Saved a trace of length {}.".format(len(raw_trace)))
                    queries += 1
                    trace = []
                    next_trace(args.feature, objectID)
            else:
                print("Can't save while recording! Please stop the recording first.")

        if replayPushes > replayNum:
            print("Replaying trace. If happy with the recording, press Save Trace; otherwise press Next Trace.")
            replayNum = replayPushes
            replay_trace(trace, objectID)

        if stopPushes > stopNum:
            print("Stopping trace recording. If happy with the recording, press Save Trace; otherwise press Next Trace.")
            stopNum = stopPushes
            record = False

            # Pre-process the recorded data.
            trace = np.squeeze(np.array(trace))
            lo = 0
            hi = trace.shape[0] - 1
            while np.linalg.norm(trace[lo] - trace[lo + 1]) < 0.01 and lo < hi:
                lo += 1
            while np.linalg.norm(trace[hi] - trace[hi - 1]) < 0.01 and hi > 0:
                hi -= 1
            trace = trace[lo:hi+1, :]
            replay_trace(trace, objectID)

        if recordPushes > recordNum:
            print("Starting trace #{} recording. Please place the robot's end-effector in a lowly-expressed feature region, and when done press Stop Recording.".format(queries+1))
            recordNum = recordPushes
            record = True

        if record:
            state = p.getJointStates(objectID["robot"], range(p.getNumJoints(objectID["robot"])))
            waypt = [s[0] for s in state]
            trace.append(waypt)

        time.sleep(0.01)

    # Save collected traces.
    i = 1
    filename = args.save_dir + "{}_{}.p".format(args.feature, i)
    while os.path.exists(filename):
        i+=1
        filename = args.save_dir + "{}_{}.p".format(args.feature, i)

    with open(filename, 'wb') as handle:
        pickle.dump(traces, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Disconnect once the session is over.
    p.disconnect()

    # Now train the feature in the background.
    LF_dict = {'bet_data':5, 'sin':False, 'cos':False, 'rpy':False, 'lowdim':False, 'norot':True,
           'noangles':True, '6D_laptop':False, '6D_human':False, '9D_coffee':False, 'EErot':False,
           'noxyz':False, 'subspace_heuristic':False}
    unknown_feature = LearnedFeature(2, 64, LF_dict)

    all_trace_data = np.empty((0, 97), float)
    for idx in range(len(traces)):
        np.flip(traces[idx],axis=0)
        trace = traces[idx]

        # Downsample for faster training if needed.
        if(trace.shape[0] > 80):
            idxes = np.random.choice(np.arange(trace.shape[0]), 100, replace=False)
            idxes = np.sort(idxes)
            idxes[0] = 0
            idxes[-1] = trace.shape[0] - 1
            trace = trace[idxes]
        unknown_feature.add_data(trace)
        all_trace_data = np.vstack((all_trace_data, trace))

    _ = unknown_feature.train(epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.)
    i = 1
    filename = "{}/{}_{}.pt".format(args.save_dir, args.feature, i)
    while os.path.exists(filename):
        i+=1
        filename = "{}/{}_{}.pt".format(args.save_dir, args.feature, i)
    torch.save(unknown_feature, filename)

    if(args.feature in ["human"]):
        # Visualize ground truth feature.
        objectID = envsetup(args, direct=True)
        viz_learned_feat("../", args.feature, objectID, traces, unknown_feature)
        p.disconnect()

