"""
Main program to run visual HTM experiments.

@author: Brody Kutt (bjk4704@rit.edu)
"""

import os
import sys
import utils
import numpy as np
from collections import OrderedDict
np.set_printoptions(threshold=np.nan)


# Usage statement
USAGE = 'USAGE: $ python run.py\n'

# Size of focus window side length in pixels
FOCUS_WIN_SIZE = 22

# For copying models
TEMP_DIR = os.path.join(os.getcwd(), 'tmp')

# All available fonts
FONTS = ['calibri', 'sans_serif', 'segoe_script', 'californian_fb', 'bell_mt']

# Number of passes through the movement cycle during training
TRAINING_ITERS = 40


def phase1(model):
    """
    Execute phase 1 of the visual HTM experiments.
    """
    # TRAINING #############################################

    # Load a single instance of 8
    im = utils.load_digit('8_calibri')
    im_focus_win = utils.ImFocusWin(im, FOCUS_WIN_SIZE)
    im_focus_win.set_move_id('vert_cascade', 5)

    hist_err = []
    # Train for 20 passes
    for i in range(20 * TRAINING_ITERS):
        # Run through HTM
        predCols = model.get_predCols()
        model.process(im_focus_win.grab_window())
        activeCols = model.get_activeCols()
        score = utils.compute_score(activeCols, predCols)
        hist_err.append(score)
        # Move three places; saccade movement is not continuous
        im_focus_win.next()
        im_focus_win.next()
        im_focus_win.next()

    # Plot the error over time while training on this digit
    utils.plot_pred_err(hist_err, 'Calibri 8')

    # TESTING ###############################################

    # Prevent anymore synapse updates
    model.stop_learning()

    # Collects avg error from all digits from all fonts
    all_avgs = {}

    # Test with instances of other digits in various fonts
    for font in FONTS:

        # Store this font's per-digit avg prediction error
        avgs = OrderedDict()
        # Iterate through each digit in this font
        for j in range(10):
            # Reset the prediction state
            model.reset_tm()
            hist_err = []
            # Load the digit instance
            im = utils.load_digit('%d_%s' % (j, font))
            im_focus_win = utils.ImFocusWin(im, FOCUS_WIN_SIZE)
            im_focus_win.set_move_id('vert_cascade', 5)

            # Test for 4 passes to get average response
            for i in range(20 * 4):
                # Run through HTM
                predCols = model.get_predCols()
                model.process(im_focus_win.grab_window())
                activeCols = model.get_activeCols()
                score = utils.compute_score(activeCols, predCols)
                # First prediction error is always going to be maximal
                if(i != 0):
                    hist_err.append(score)
                # Move three places; saccade movement is not continuous
                im_focus_win.next()
                im_focus_win.next()
                im_focus_win.next()

            avgs[str(j)] = np.mean(hist_err)

        # Collect to overall container
        all_avgs[font] = avgs

    # Plot the average error per digit per font
    utils.plot_avgs(all_avgs, 'Calibri 8')


def phase2(model):
    """
    Execute phase 2 of the visual HTM experiments.
    """
    # TRAINING #############################################

    # First train up the HTM model to the same state it used to be in:
    # Load a single instance of 8
    im = utils.load_digit('8_calibri')
    im_focus_win = utils.ImFocusWin(im, FOCUS_WIN_SIZE)
    im_focus_win.set_move_id('vert_cascade', 5)

    hist_err = []
    # Train for 20 passes
    for i in range(20 * TRAINING_ITERS):
        # Run through HTM
        predCols = model.get_predCols()
        model.process(im_focus_win.grab_window())
        activeCols = model.get_activeCols()
        score = utils.compute_score(activeCols, predCols)
        hist_err.append(score)
        # Move three places; saccade movement is not continuous
        im_focus_win.next()
        im_focus_win.next()
        im_focus_win.next()

    # Now train it on an instance of a different digit:
    # Reset the prediction state
    model.reset_tm()

    # Load a single instance of 7
    im = utils.load_digit('7_calibri')
    im_focus_win = utils.ImFocusWin(im, FOCUS_WIN_SIZE)
    im_focus_win.set_move_id('vert_cascade', 5)

    hist_err = []
    # Train for 20 passes
    for i in range(20 * TRAINING_ITERS):
        # Run through HTM
        predCols = model.get_predCols()
        model.process(im_focus_win.grab_window())
        activeCols = model.get_activeCols()
        score = utils.compute_score(activeCols, predCols)
        hist_err.append(score)
        # Move three places; saccades movement is not continuous
        im_focus_win.next()
        im_focus_win.next()
        im_focus_win.next()

    # Now train it on an instance of a different digit:
    # Reset the prediction state
    model.reset_tm()

    # Load a single instance of 2
    im = utils.load_digit('2_calibri')
    im_focus_win = utils.ImFocusWin(im, FOCUS_WIN_SIZE)
    im_focus_win.set_move_id('vert_cascade', 5)

    hist_err = []
    # Train for 20 passes
    for i in range(20 * TRAINING_ITERS):
        # Run through HTM
        predCols = model.get_predCols()
        model.process(im_focus_win.grab_window())
        activeCols = model.get_activeCols()
        score = utils.compute_score(activeCols, predCols)
        hist_err.append(score)
        # Move three places; saccades movement is not continuous
        im_focus_win.next()
        im_focus_win.next()
        im_focus_win.next()

    # TESTING ###############################################

    # Do the exact same testing procedure as Phase 1

    # Prevent anymore synapse updates
    model.stop_learning()

    # Collects avg error from all digits from all fonts
    all_avgs = {}

    # Test with instances of other digits in various fonts
    for font in FONTS:

        # Store this font's per-digit avg prediction error
        avgs = OrderedDict()
        # Iterate through each digit in this font
        for j in range(10):
            # Reset the prediction state
            model.reset_tm()
            hist_err = []
            # Load the digit instance
            im = utils.load_digit('%d_%s' % (j, font))
            im_focus_win = utils.ImFocusWin(im, FOCUS_WIN_SIZE)
            im_focus_win.set_move_id('vert_cascade', 5)

            # Test for 4 passes to get average response
            for i in range(20 * 4):
                # Run through HTM
                predCols = model.get_predCols()
                model.process(im_focus_win.grab_window())
                activeCols = model.get_activeCols()
                score = utils.compute_score(activeCols, predCols)
                # First prediction error is always going to be maximal
                if(i != 0):
                    hist_err.append(score)
                # Move three places; saccade movement is not continuous
                im_focus_win.next()
                im_focus_win.next()
                im_focus_win.next()

            avgs[str(j)] = np.mean(hist_err)

        # Collect to overall container
        all_avgs[font] = avgs

    # Plot the average error per digit per font
    utils.plot_avgs(all_avgs, 'Calibri 8 then 7 then 2')


if __name__ == '__main__':
    if(len(sys.argv) > 1):
        print 'Err: No arguments expected.\n'
        print USAGE
        sys.exit()
    model = utils.init_htm(FOCUS_WIN_SIZE)
    # phase1(model)
    phase2(model)
