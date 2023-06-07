import numpy as np
import glob, os

def get_redshifts_from_output(output_dir, z_low=None, z_high=None, bracket=False):
    output_files = glob.glob(os.path.join(output_dir,'xfrac*'))

    redshifts = []
    for f in output_files:
        try:
            z = float(f.split('_')[-1][:-4])
            redshifts.append(z)
        except: 
            pass
    
    return np.array(redshifts)

def find_bins(input_array, binning_array):
    # Sort the binning array in ascending order
    sorted_bins = np.sort(binning_array)

    left_bins = []
    right_bins = []

    if isinstance(input_array, (np.ndarray, list)):
        for value in input_array:
            # Find the index where the value should be inserted in the sorted_bins
            bin_index = np.digitize(value, sorted_bins)

            # Check if bin_index is within the bounds of the sorted_bins
            if bin_index > 0 and bin_index <= len(sorted_bins):
                left_bin = sorted_bins[bin_index - 1]
                right_bin = sorted_bins[bin_index]
            elif bin_index == 0:
                left_bin = None
                right_bin = sorted_bins[bin_index]
            else:
                left_bin = sorted_bins[bin_index - 1]
                right_bin = None

            left_bins.append(left_bin)
            right_bins.append(right_bin)
        return np.array(left_bins), np.array(right_bins)

    else:
        value = input_array
        # Find the index where the value should be inserted in the sorted_bins
        bin_index = np.digitize(value, sorted_bins)

        # Check if bin_index is within the bounds of the sorted_bins
        if bin_index > 0 and bin_index <= len(sorted_bins):
            left_bin = sorted_bins[bin_index - 1]
            right_bin = sorted_bins[bin_index]
        elif bin_index == 0:
            left_bin = None
            right_bin = sorted_bins[bin_index]
        else:
            left_bin = sorted_bins[bin_index - 1]
            right_bin = None

        left_bins.append(left_bin)
        right_bins.append(right_bin)

        return left_bins[0], right_bins[0]
