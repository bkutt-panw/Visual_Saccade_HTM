"""
Utilities used to run visual HTM experiments.

@author: Brody Kutt (bjk4704@rit.edu)
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps
# from nupic.algorithms.spatial_pooler import SpatialPooler  # Python
from nupic.bindings.algorithms import SpatialPooler  # CPP
from nupic.algorithms.backtracking_tm_cpp import BacktrackingTMCPP as TM  # CPP

# Folder containing digit pngs
IM_DIR_PATH = os.path.join(os.getcwd(), 'digits')
IM_FILE_TYPE = '.png'

# Some plotting properties
TICK_FNT_SZ = 8
AX_LBL_FNT_SZ = 12
TITLE_FNT_SZ = 15
SAVE_FIGS = False


def plot_pred_err(hist_errs, plt_title):
    """
    Create the timestep error plot.
    """
    fig, ax = plt.subplots()
    ax.set_title('Prediction Errors for ' + plt_title,
                 fontsize=TITLE_FNT_SZ,
                 fontweight='bold')
    x = np.arange(0, len(hist_errs))

    # Create first-order timestep error plot
    ax.plot(x, hist_errs, 'b')
    ax.fill_between(x, 0, hist_errs, facecolor='b')
    ax.grid()
    ax.set_xlabel('Timestep', fontsize=AX_LBL_FNT_SZ, fontweight='bold')
    ax.set_ylabel('Prediction Error', fontsize=AX_LBL_FNT_SZ, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(hist_errs) - 1])

    # Finish up and save
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig('pred_errs.png')
    else:
        plt.show()


def plot_avgs(all_avgs, plt_title):
    """
    Create the average timestep error figure.
    """
    gs = gridspec.GridSpec(3, 4)
    # gs.update(wspace=0.5)
    plt.figure(figsize=(7, 10))
    plt.suptitle('Average Prediction Error Per-Digit\nAfter Training on ' + plt_title,
                 fontsize=20,
                 fontweight='bold')

    ax1 = plt.subplot(gs[0, :2])  # row 0, col 0
    ax1.set_title('Calibri')
    D = all_avgs['calibri']
    ax1.bar(range(len(D)), list(D.values()), align='center')
    ax1.set_xticks(range(len(D)))
    ax1.set_xticklabels(list(D.keys()))
    ax1.yaxis.grid()  # horizontal lines

    ax2 = plt.subplot(gs[0, 2:])  # row 0, col 1
    ax2.set_title('Sans Serif')
    D = all_avgs['sans_serif']
    ax2.bar(range(len(D)), list(D.values()), align='center')
    ax2.set_xticks(range(len(D)))
    ax2.set_xticklabels(list(D.keys()))
    ax2.yaxis.grid()  # horizontal lines

    ax3 = plt.subplot(gs[1, :2])  # row 1, col 0
    ax3.set_title('Segoe Script')
    D = all_avgs['segoe_script']
    ax3.bar(range(len(D)), list(D.values()), align='center')
    ax3.set_xticks(range(len(D)))
    ax3.set_xticklabels(list(D.keys()))
    ax3.yaxis.grid()  # horizontal lines

    ax4 = plt.subplot(gs[1, 2:])  # row 1, col 1
    ax4.set_title('Californian FB')
    D = all_avgs['californian_fb']
    ax4.bar(range(len(D)), list(D.values()), align='center')
    ax4.set_xticks(range(len(D)))
    ax4.set_xticklabels(list(D.keys()))
    ax4.yaxis.grid()  # horizontal lines

    ax5 = plt.subplot(gs[2, 1:3])  # row 2, span all columns
    ax5.set_title('Bell MT')
    D = all_avgs['bell_mt']
    ax5.bar(range(len(D)), list(D.values()), align='center')
    ax5.set_xticks(range(len(D)))
    ax5.set_xticklabels(list(D.keys()))
    ax5.yaxis.grid()  # horizontal lines

    # Finish up and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if SAVE_FIGS:
        plt.savefig('pred_errs.png')
    else:
        plt.show()


def init_htm(focus_win_size):
    """
    Initialize the HTM model for a given focus window size.
    """
    # Initialize the Spatial Pooler instance
    sp = SpatialPooler(inputDimensions=[focus_win_size * focus_win_size],
                       columnDimensions=[2048],
                       synPermInactiveDec=0.0001,
                       synPermConnected=0.4,
                       synPermActiveInc=0.008,
                       potentialRadius=3,
                       potentialPct=0.8,
                       numActiveColumnsPerInhArea=40,
                       boostStrength=0.0,
                       globalInhibition=1,
                       seed=1956)

    # Initialize the Temporal Memory instance
    tm = TM(numberOfCols=2048,
            cellsPerColumn=32,
            initialPerm=0.21,
            minThreshold=10,
            permanenceInc=0.1,
            permanenceDec=0.1,
            activationThreshold=15,
            maxSegmentsPerCell=128,
            maxSynapsesPerSegment=32,
            newSynapseCount=20,
            globalDecay=0.0,
            maxAge=0,
            pamLength=3,
            seed=1960)

    return HTM_Model(sp, tm)


def load_digit(name, show=False):
    """
    Grabs the PIL Image instance of an image. The image is converted from RGB
    to binary.

    @param name = name of the digit file
    """
    im_path = os.path.join(IM_DIR_PATH, name + IM_FILE_TYPE)
    im = Image.open(im_path).convert('L')
    im = ImageOps.invert(im)
    im = im.convert('1')
    if(show):
        im.show()
    return np.array(im, dtype=np.uint32)


def compute_score(sp_sdr, pred_sdr):
    """
    Computes the prediction error score. This is the fraction of active
    columns not predicted.
    """
    nActiveColumns = len(sp_sdr)
    if nActiveColumns > 0:
        # Sum of total # of columns that are active and were predicted.
        score = np.in1d(sp_sdr, pred_sdr).sum()
        # Percent of active columns that were not predicted
        score = (nActiveColumns - score) / float(nActiveColumns)
    else:
        score = 0.0  # There are no active columns.
    return score


class HTM_Model(object):
    """
    Class that manages a spatial pooler and temporal memory instance.
    """
    def __init__(self, sp, tm):
        self.sp = sp
        self.tm = tm
        self.activeCols = np.zeros(tm.numberOfCols, dtype="float32")
        self.predCols = np.zeros(tm.numberOfCols, dtype="float32")
        self.learn = True

    def get_sp(self):
        """
        Return the SP instance.
        """
        return self.sp

    def get_tm(self):
        """
        Return the TM instance.
        """
        return self.tm

    def get_activeCols(self):
        """
        Return the output of SP.
        """
        return self.activeCols.nonzero()[0]

    def get_predCols(self):
        """
        Return the output of TM.
        """
        return self.predCols.nonzero()[0]

    def stop_learning(self):
        """
        Turn learning off for both SP and TM
        """
        self.learn = False

    def start_learning(self):
        """
        Turn learning back on for both SP and TM
        """
        self.learn = True

    def reset_tm(self):
        """
        Reset the learning state.
        """
        self.tm.reset()

    def process(self, window):
        """
        Run a new focus window through the HTM model.
        """
        # Run the focus window through the SP
        self.sp.compute(window, self.learn, self.activeCols)
        # Run TM algorithm (inference is always turned on)
        self.tm.compute(self.activeCols, self.learn, True)
        # Compute the predicted columns
        self.predCols = self.tm.topDownCompute()


class ImFocusWin(object):
    """
    Class that manages an image with a movable square focus window inside of it.
    """
    def __init__(self, im, win_size, verbosity=0):
        """
        The focus window is initialized to the center of the image.
        """
        assert win_size > 0
        assert win_size <= im.shape[0] and win_size <= im.shape[1]
        self.im = im
        self.win_size = win_size
        self.verbosity = verbosity

        # Define the focus window location for the center of image
        self.x_centered = [int((im.shape[1] - win_size) / 2),
                           int((im.shape[1] - win_size) / 2) + win_size - 1]
        self.y_centered = [int((im.shape[0] - win_size) / 2),
                           int((im.shape[0] - win_size) / 2) + win_size - 1]

        # Current location of focus window (initialized to center)
        self.x_range = copy.deepcopy(self.x_centered)
        self.y_range = copy.deepcopy(self.y_centered)

        # Additional data for predefined movement behavior
        self.move_id = None
        self.move_info = None
        self.ref_x = None
        self.ref_y = None
        self.move_vec = None
        self.move_ptr = None

    def recenter(self):
        """
        Bring the focus window back to the center of the image.
        """
        self.x_range = copy.deepcopy(self.x_centered)
        self.y_range = copy.deepcopy(self.y_centered)

    def set_move_id(self, move_id, move_info):
        """
        Set up a new behavior of focus window movement from current location.
        Returns true upon successful completion.
        """
        self.move_id = move_id
        self.move_info = move_info

        if(move_id == 'vert_cascade'):
            # Move the focus window top to bottom. The move_info parameter
            # defines the distance in either direction from the reference
            # it moves before switching directions.
            y_mov = np.concatenate((np.ones(move_info),
                                   -1 * np.ones(2 * move_info),
                                   np.ones(move_info)))
            self.move_vec = np.hstack((np.zeros((4 * move_info, 1)),
                                      y_mov.reshape(y_mov.size, 1)))

        elif(move_id == 'horiz_cascade'):
            # Move the focus window left to right. The move_info parameter
            # defines the distance in either direction from the reference
            # it moves before switching directions.
            x_mov = np.concatenate((np.ones(move_info),
                                   -1 * np.ones(2 * move_info),
                                   np.ones(move_info)))
            self.move_vec = np.hstack((x_mov.reshape(x_mov.size, 1),
                                      np.zeros((4 * move_info, 1))))

        elif(move_id == 'random'):
            # Move a random perturbation from the reference point. The move_info
            # parameter defines the maximum possible distance from reference in
            # any direction.
            self.ref_x = copy.deepcopy(self.x_range)
            self.ref_y = copy.deepcopy(self.y_range)

        else:
            print 'Movement ID not recognized.'
            assert False

        self.move_ptr = 0

    def next(self):
        """
        Move the focus window to the next position given its current position
        and the specified move behavior id. Returns true upon successful
        movement.
        """
        if(self.move_id is None):
            print 'Movement behavior has not been specified.'
            return False

        if(self.move_id == 'random'):
            # First return to reference point
            self.x_range = copy.deepcopy(self.ref_x)
            self.y_range = copy.deepcopy(self.ref_y)

            # TODO

            return False

        elif(self.move_id == 'vert_cascade' or self.move_id == 'horiz_cascade'):
            move_instructions = self.move_vec[self.move_ptr]

            # Move in x direction
            if(move_instructions[0] != 0):
                if(move_instructions[0] < 0):
                    self.move('left', -1 * move_instructions[0])
                else:
                    self.move('right', move_instructions[0])

            # Move in y direction
            if(move_instructions[1] != 0):
                if(move_instructions[1] < 0):
                    self.move('down', -1 * move_instructions[1])
                else:
                    self.move('up', move_instructions[1])

            # Advance the pointer
            self.move_ptr = (self.move_ptr + 1) % len(self.move_vec)
            return True

        return False

    def move(self, direction, dist):
        """
        Move the focus window a specified distance and direction. If the
        distance goes outside the bounds of the image, it will stop at the edge.

        @param dist = integer distance in pixels to move
        @param dir = direction of movement either 'up', 'left', 'right' or 'down'
        """
        assert dist >= 0
        if(direction == 'up'):
            new_top = max((self.y_range[0] - dist), 0)
            if (new_top == 0 and self.verbosity != 0): print 'Top limit reached.'
            self.y_range = [new_top, new_top + self.win_size - 1]
        elif(direction == 'left'):
            new_left = max((self.x_range[0] - dist), 0)
            if (new_left == 0 and self.verbosity != 0): print 'Left limit reached.'
            self.x_range = [new_left, new_left + self.win_size - 1]
        elif(direction == 'right'):
            new_right = min((self.x_range[1] + dist), self.im.shape[1] - 1)
            if (new_right == self.im.shape[1] - 1 and self.verbosity != 0): print 'Right limit reached.'
            self.x_range = [new_right - (self.win_size - 1), new_right]
        elif(direction == 'down'):
            new_bottom = min((self.y_range[1] + dist), self.im.shape[0] - 1)
            if (new_bottom == self.im.shape[0] - 1 and self.verbosity != 0): print 'Bottom limit reached.'
            self.y_range = [new_bottom - (self.win_size - 1), new_bottom]
        else:
            print 'Uninterpretable direction encountered.'

    def grab_window(self, flatten=True):
        """
        Grab the slice of the image that the focus window is currently
        positioned over. Returns in numpy format.
        """
        window = self.im[int(self.y_range[0]):int(self.y_range[1] + 1),
                         int(self.x_range[0]):int(self.x_range[1] + 1)]
        return window.flatten() if flatten else window

    def show_window(self):
        """
        Show where in the image the current focus window is oriented.

        TODO -
        draw a red box around focus window on original image instead.
        """
        window = self.im[int(self.y_range[0]):int(self.y_range[1] + 1),
                         int(self.x_range[0]):int(self.x_range[1] + 1)]
        im = Image.fromarray(np.uint8(window * 255), 'L')
        im.show()
