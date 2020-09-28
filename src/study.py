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


if __name__ == "__main__":
    # Parse experimental arguments.
    args = parse_arguments()

    # Connect to physics simulator.
    physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version

    # Add path to data resources for the environment.
    p.setAdditionalSearchPath(args.resources_dir)

    # Setup the environment.
    objectID = setup_environment()

    # Get rid of gravity and make simulation happen in real time.
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(1)

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

    # Visualize ground truth feature.
    viz_gt_feature("../", args.feature, objectID)

    N_QUERIES = 10
    trace = []
    traces = []
    start_labels = [1.0] * N_QUERIES
    end_labels = [0.0] * N_QUERIES
    queries = 0
    record = False
    print("Attempting trace #{}. Please place the robot's end-effector in a highly-expressed feature region. Explore the world, use given camera views if needed, and when ready press Start Recording.".format(queries+1))
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
            p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-89.9,
                                         cameraPitch=-89.9,
                                         cameraTargetPosition=[0.0,0.0,0.0])

        if top2Pushes > top2Num:
            print("Changing the view to top-down #2.")
            top2Num = top2Pushes
            p.resetDebugVisualizerCamera(cameraDistance=2.3, cameraYaw=-179.9,
                                         cameraPitch=-89.9,
                                         cameraTargetPosition=[-0.2,0,0.2])

        if sidePushes > sideNum:
            print("Changing the view to side.")
            sideNum = sidePushes
            p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-270,
                                         cameraPitch=-15,
                                         cameraTargetPosition=[0,0,0.15])

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
            print("Attempting trace #{}. Please place the robot's end-effector in a highly-expressed feature region. Explore the world, use given camera views if needed, and when ready press Start Recording.".format(queries+1))
            nextNum = nextPushes
            trace = []
            move_robot(objectID["robot"])

        if savePushes > saveNum:
            saveNum = savePushes
            if record == False:
                # Pre-process the recorded data before training.
                trace = np.squeeze(np.array(trace))
                lo = 0
                hi = trace.shape[0] - 1
                while np.linalg.norm(trace[lo] - trace[lo + 1]) < 0.01 and lo < hi:
                    lo += 1
                while np.linalg.norm(trace[hi] - trace[hi - 1]) < 0.01 and hi > 0:
                    hi -= 1
                trace = trace[lo:hi+1, :]

                raw_trace = []
                for waypt in trace:
                    raw_trace.append(raw_features(objectID, waypt))

                # Save trace.
                raw_trace.reverse()
                traces.append(np.array(raw_trace))
                print("Saved a trace of length {}.".format(len(raw_trace)))
                trace = []
                queries += 1
            else:
                print("Can't save while recording! Please stop the recording first.")

        if replayPushes > replayNum:
            print("Replaying trace. Feel free to change the view and replay the trace again using the Replay Trace button. If happy with the recording, press Save Trace; otherwise press Next Trace.")
            replayNum = replayPushes
            for waypt in trace:
                for jointIndex in range(p.getNumJoints(objectID["robot"])):
                    p.resetJointState(objectID["robot"], jointIndex, waypt[jointIndex])
                time.sleep(0.01)

        if stopPushes > stopNum:
            print("Stopping trace recording. Feel free to change the view and replay the trace using the Replay Trace button. If happy with the recording, press Save Trace; otherwise press Next Trace.")
            stopNum = stopPushes
            record = False

        if recordPushes > recordNum:
            print("Starting trace #{} recording. Please place the robot's end-effector in a lowly-expressed feature region. Explore the world, use given camera views if needed, and when done press Stop Recording.".format(queries+1))
            recordNum = recordPushes
            record = True

        if record:
            state = p.getJointStates(objectID["robot"], range(p.getNumJoints(objectID["robot"])))
            waypt = [s[0] for s in state]
            trace.append(waypt)

        time.sleep(0.01)

    # Save collected traces.
    filename = args.save_dir + "{}.p".format(args.feature)
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
        unknown_feature.add_data(traces[idx])
        all_trace_data = np.vstack((all_trace_data, traces[idx]))

    _ = unknown_feature.train(epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.)
    torch.save(unknown_feature, '{}/{}.pt'.format(args.save_dir, args.feature))
