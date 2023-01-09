######################################
###    AUTHOR : MATHIEU HUSSONG    ###
###   ENAC TELECOM LAB COPYRIGHT   ###
###        DATE : 16/02/2022       ###
###          VERSION : 1           ###
######################################

# THIS CODE INTERPOLATES THE POSITION AND THE TIME BIAS OF THE SATELLITES BETWEEN KNOWN POSITION AND TIME


#######################
### LIBRARY IMPORTS ###
#######################

import numpy as np
import sys
import conversion
if sys.platform == "win32":
    import matplotlib.pyplot as plt


#################
### CONSTANTS ###
#################

C = 299792458


#################
### FUNCTIONS ###
#################

def least_square_interpolation(satellite_positions, epoch, single_sat="", interp_degree=10, points_of_interest=20,
                               recursion_bool=True):
    """This function interpolates the satellite positions between known positions and time.
    The interpolation method consists in a weighted least square estimation of a satellite polynomial trajectory.
    INPUTS:
        satellite_positions: the SatPosition structure containing the data from sp3 files
        epoch :  the epoch at which to interpolate the satellite positions (in GPS time)
        single_sat : specify a single satellite of which to interpolate the position (as a string in the style 'G01')
                     leave single_sat="" if you want to interpolate the position of every satellites
        interp_degree : the degree of the polynomial that matches the satellite trajectory (default = 10)
        points_of_interest : the number of satellite known locations to use for interpolation (default = 20)
        recursion_bool : True if the function needs to be recursively called to improve the accuracy, False otherwise
                         please do not modify this argument, unless you know what you're doing
    OUTPUTS:
        satellite_interpolated_positions : a dictionary with the interpolated satellite positions at the input epoch
        satellite_interpolated_velocities: a dictionary with the interpolated satellite velocities at the input epoch"""
    # FINDING THE APPROPRIATE KNOWN EPOCHS TO CONSIDER TO INTERPOLATE THE TRAJECTORY AT THE EPOCH 'epoch'
    start_of_epochs_of_interest = 0

    while epoch > satellite_positions.epochs[start_of_epochs_of_interest]["time"] \
            and start_of_epochs_of_interest < satellite_positions.nb_epochs - points_of_interest/2:
        start_of_epochs_of_interest += 1
    start_of_epochs_of_interest = max(0, start_of_epochs_of_interest - int(points_of_interest / 2))

    # AT THIS MOMENT, WE KNOW WHICH SAMPLES FROM THE SP3 FLE TO CONSIDER TO INTERPOLATE THE POSITION FROM
    # INDEED, THE VARIABLE start_of_epochs_of_interest CONTAINS THE FIRST EPOCH OF THE CONSIDERED SAMPLES
    # THE N OTHER SAMPLES TO CONSIDER ARE THE N SAMPLES THAT FOLLOWS THE ONE AT EPOCH=start_of_epochs_of_interest

    satellite_interpolated_positions = {}
    for sat, sat_position in satellite_positions.epochs[start_of_epochs_of_interest].items():
        if sat != "time" and (single_sat == "" or single_sat == sat):

            # CREATION OF THE MATRICES FOR LSE
            sat_pos_x = np.zeros((points_of_interest, 1))
            sat_pos_y = np.zeros((points_of_interest, 1))
            sat_pos_z = np.zeros((points_of_interest, 1))
            sat_pos_time = np.zeros((points_of_interest, 1))
            for i in range(points_of_interest):
                sat_pos_x[i] = satellite_positions.epochs[start_of_epochs_of_interest + i][sat][0]
                sat_pos_y[i] = satellite_positions.epochs[start_of_epochs_of_interest + i][sat][1]
                sat_pos_z[i] = satellite_positions.epochs[start_of_epochs_of_interest + i][sat][2]
                sat_pos_time[i] = satellite_positions.epochs[start_of_epochs_of_interest + i][sat][3]

            coeffs = np.zeros((points_of_interest, interp_degree))
            for i in range(points_of_interest):
                for j in range(interp_degree):
                    coeffs[i, j] = (satellite_positions.epochs[start_of_epochs_of_interest + i]["time"] - epoch) ** j
            coeffs_time = np.zeros((points_of_interest, 2))
            for i in range(points_of_interest):
                for j in range(2):
                    coeffs_time[i, j] = (satellite_positions.epochs[start_of_epochs_of_interest + i]["time"] - epoch)**j

            weight = np.zeros((points_of_interest, points_of_interest))
            for i in range(points_of_interest):
                weight[i, i] = 1 / (1 + abs(satellite_positions.epochs[start_of_epochs_of_interest + i]["time"] - epoch))

            # DETERMINATION OF THE INTERPOLATION POLYNOMIAL BY COMPUTING THE LSE SOLUTION
            interp_poly_coefficients_x = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(coeffs), weight),
                                                coeffs)), np.transpose(coeffs)), weight), sat_pos_x)
            interp_poly_coefficients_y = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(coeffs), weight),
                                                coeffs)), np.transpose(coeffs)), weight), sat_pos_y)
            interp_poly_coefficients_z = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(coeffs), weight),
                                                coeffs)), np.transpose(coeffs)), weight), sat_pos_z)
            interp_poly_coefficients_time = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(coeffs_time),
                                            weight), coeffs_time)), np.transpose(coeffs_time)), weight), sat_pos_time)

            # INTERPOLATION OF THE POSITION AT THE EPOCH 'epoch'
            # AS WE DEFINED OUR TIME ORIGIN AS THE EPOCH "epoch', THE SOLUTION IS THE CONSTANT POLYNOMIAL TERM
            sat_interpolated_x = interp_poly_coefficients_x[0]
            sat_interpolated_y = interp_poly_coefficients_y[0]
            sat_interpolated_z = interp_poly_coefficients_z[0]
            sat_interpolated_time = interp_poly_coefficients_time[0]

            satellite_interpolated_positions[sat] = np.array((sat_interpolated_x[0], sat_interpolated_y[0],
                                                              sat_interpolated_z[0], sat_interpolated_time[0]))

    if recursion_bool:
        satellite_interpolated_velocities = {}
        # TO COMPUTE THE VELOCITY, WE DERIVE THE SATELLITE POSITIONS AT EPOCH = epoch - 1 second.
        # AND THEN WE USE THE APPROXIMATION THAT V = D / T
        satellite_interpolated_previous_positions = least_square_interpolation(satellite_positions, epoch-1,
                                                        single_sat, interp_degree, points_of_interest, False)
        for sat, sat_position in satellite_positions.epochs[start_of_epochs_of_interest].items():
            if sat != "time" and (single_sat == "" or single_sat == sat):
                satellite_interpolated_velocities[sat] = satellite_interpolated_positions[sat][0:3] - \
                                                         satellite_interpolated_previous_positions[sat][0:3]

        return satellite_interpolated_positions, satellite_interpolated_velocities

    return satellite_interpolated_positions


def neville_interpolation(satellite_positions, epoch, single_sat="", leap_seconds=18):
    """this function interpolates the satellite positions between known positions and time.
    The interpolation method consists in a Neville interpolation of the satellite trajectories.
    INPUTS:
        satellite_positions: the SatPosition structure containing the data from sp3 files
        epoch :  the epoch at which to interpolate the satellite positions (in GPS time)
        single_sat : specify a single satellite of which to interpolate the position (as a string in the style 'G01')
                     leave single_sat="" if you want to interpolate the position of every satellites
        leap_seconds : the number of GPS leap seconds at the time the interpolation is performed (default=18)
    OUTPUTS:
        satellite_interpolated_positions : a dictionary with the interpolated satellite positions at the input epoch
        satellite_interpolated_velocities: a dictionary with the interpolated satellite velocities at the input epoch"""
    satellite_interpolated_positions = {}
    satellite_interpolated_velocities = {}

    # FOR EVERY SATELLITE POSITION TO INTERPOLATE, WE RUN A NEVILLE INTERPOLATION METHOD
    for sat, sat_position in satellite_positions.epochs[0].items():
        if sat != "time" and (single_sat == "" or single_sat == sat):
            # tr_time represents the number of seconds since midnight
            tr_time = (epoch - leap_seconds) % 86400
            sat_position, sat_velocity, _ = neville(satellite_positions, sat, tr_time)
            satellite_interpolated_positions[sat] = sat_position
            satellite_interpolated_velocities[sat] = sat_velocity

    return satellite_interpolated_positions, satellite_interpolated_velocities


def neville(satellite_positions, id_sat, tr_time):
    """this function executes a minimization resolution based on the Neville algorithm"""

    sat_psinter_nump = 11  # interpolation degree for the position
    sat_csckinter_nump = 3  # interpolation degree for the clock offset
    Precise_time = np.linspace(0, satellite_positions.nb_epochs * 300, satellite_positions.nb_epochs + 1)
    inter_sec_delta = 300

    time_interval_ind = int(max(0, np.floor(tr_time / inter_sec_delta)))
    Xdata_comp = [float(tr_time)]

    # X estimation

    y_absp = []
    for i in range(satellite_positions.nb_epochs):
        y_absp.append(satellite_positions.epochs[i][id_sat][0])

    if time_interval_ind + sat_psinter_nump > len(Precise_time) - 1:
        x_abs = Precise_time[time_interval_ind - sat_psinter_nump:time_interval_ind + 1]
        y_abs = y_absp[time_interval_ind - sat_psinter_nump:time_interval_ind + 1]

    else:
        if not time_interval_ind:
            x_abs = Precise_time[time_interval_ind:time_interval_ind + sat_psinter_nump + 1]
            y_abs = y_absp[time_interval_ind:time_interval_ind + sat_psinter_nump + 1]
        else:
            x_abs = Precise_time[time_interval_ind - 1:time_interval_ind + sat_psinter_nump]
            y_abs = y_absp[time_interval_ind - 1:time_interval_ind + sat_psinter_nump]

    y_inter = np.zeros((len(Xdata_comp), 1))
    yveli_inter = np.zeros((len(Xdata_comp), 1))

    n = len(x_abs)

    for k in range(len(Xdata_comp)):
        xd = np.zeros(n)
        for i in range(n):
            xd[i] = abs(x_abs[i] - Xdata_comp[k])
        i = sorted(range(len(xd)), key=lambda m: xd[m])

        x_abs_sorted = []
        y_abs_sorted = []
        for ii in range(len(i)):
            x_abs_sorted.append(x_abs[i[ii]])
            y_abs_sorted.append(y_abs[i[ii]])

        P = np.zeros((n, n))
        P[:,0] = y_abs

        for i in range(n-1):
            for j in range(n-i-1):
                P[j, i + 1] = ((Xdata_comp[k] - x_abs[j]) * P[j + 1, i] + (x_abs[j+i+1]- Xdata_comp[k]) * P[j, i]) / (
                x_abs[j+i+1] - x_abs[j])

        y_inter[k] = P[0, n-1]

        D = np.zeros((n, n))
        D[:, 0] = y_abs

        for i in range(n-1):
            D[i, 1] = (D[i+1, 0] - D[i, 0]) / (x_abs[i+1] - x_abs[i])

        for i in range(1, n):
            for j in range(n-i-1):
                D[j, i + 1] = (P[j + 1, i]+ (Xdata_comp[k] - x_abs[j]) * D[j + 1, i] - P[j, i] + (
                x_abs[j + i + 1] - Xdata_comp[k]) * D[j, i]) / (x_abs[j + i + 1] - x_abs[j])

        yveli_inter[k] = D[0, n-1]

    X_inter = y_inter[0][0]
    VX_inter = yveli_inter[0][0]

    # Y estimation

    y_absp = []
    for i in range(satellite_positions.nb_epochs):
        y_absp.append(satellite_positions.epochs[i][id_sat][1])

    if time_interval_ind + sat_psinter_nump > len(Precise_time) - 1:
        x_abs = Precise_time[time_interval_ind - sat_psinter_nump + 1:time_interval_ind + 2]
        y_abs = y_absp[time_interval_ind - sat_psinter_nump + 1:time_interval_ind + 2]

    else:
        if not time_interval_ind:
            x_abs = Precise_time[time_interval_ind + 1:time_interval_ind + sat_psinter_nump + 2]
            y_abs = y_absp[time_interval_ind + 1:time_interval_ind + sat_psinter_nump + 2]
        else:
            x_abs = Precise_time[time_interval_ind:time_interval_ind + sat_psinter_nump + 1]
            y_abs = y_absp[time_interval_ind:time_interval_ind + sat_psinter_nump + 1]

    y_inter = np.zeros((len(Xdata_comp), 1))
    yveli_inter = np.zeros((len(Xdata_comp), 1))

    n = len(x_abs)

    for k in range(len(Xdata_comp)):
        xd = np.zeros(n)
        for i in range(n):
            xd[i] = abs(x_abs[i] - Xdata_comp[k])
        i = sorted(range(len(xd)), key=lambda m: xd[m])

        x_abs_sorted = []
        y_abs_sorted = []
        for ii in range(len(i)):
            x_abs_sorted.append(x_abs[i[ii]])
            y_abs_sorted.append(y_abs[i[ii]])

        P = np.zeros((n, n))
        P[:, 0] = y_abs

        for i in range(n - 1):
            for j in range(n - i - 1):
                P[j, i + 1] = ((Xdata_comp[k] - x_abs[j]) * P[j + 1, i] + (x_abs[j + i + 1] - Xdata_comp[k]) * P[
                    j, i]) / (
                                      x_abs[j + i + 1] - x_abs[j])

        y_inter[k] = P[0, n - 1]

        D = np.zeros((n, n))
        D[:, 0] = y_abs

        for i in range(n - 1):
            D[i, 1] = (D[i + 1, 0] - D[i, 0]) / (x_abs[i + 1] - x_abs[i])

        for i in range(1, n - 1):
            for j in range(n - i - 1):
                D[j, i + 1] = (P[j + 1, i] + (Xdata_comp[k] - x_abs[j]) * D[j + 1, i] - P[j, i] + (
                        x_abs[j + i + 1] - Xdata_comp[k]) * D[j, i]) / (x_abs[j + i + 1] - x_abs[j])

        yveli_inter[k] = D[0, n - 1]

    Y_inter = y_inter[0][0]
    VY_inter = yveli_inter[0][0]

    # Z estimation

    y_absp = []
    for i in range(satellite_positions.nb_epochs):
        y_absp.append(satellite_positions.epochs[i][id_sat][2])

    if time_interval_ind + sat_psinter_nump > len(Precise_time) - 1:
        x_abs = Precise_time[time_interval_ind - sat_psinter_nump + 1:time_interval_ind + 2]
        y_abs = y_absp[time_interval_ind - sat_psinter_nump + 1:time_interval_ind + 2]

    else:
        if not time_interval_ind:
            x_abs = Precise_time[time_interval_ind + 1:time_interval_ind + sat_psinter_nump + 2]
            y_abs = y_absp[time_interval_ind + 1:time_interval_ind + sat_psinter_nump + 2]
        else:
            x_abs = Precise_time[time_interval_ind:time_interval_ind + sat_psinter_nump + 1]
            y_abs = y_absp[time_interval_ind:time_interval_ind + sat_psinter_nump + 1]


    y_inter = np.zeros((len(Xdata_comp), 1))
    yveli_inter = np.zeros((len(Xdata_comp), 1))

    n = len(x_abs)

    for k in range(len(Xdata_comp)):
        xd = np.zeros(n)
        for i in range(n):
            xd[i] = abs(x_abs[i] - Xdata_comp[k])
        i = sorted(range(len(xd)), key=lambda m: xd[m])

        x_abs_sorted = []
        y_abs_sorted = []
        for ii in range(len(i)):
            x_abs_sorted.append(x_abs[i[ii]])
            y_abs_sorted.append(y_abs[i[ii]])

        P = np.zeros((n, n))
        P[:, 0] = y_abs

        for i in range(n - 1):
            for j in range(n - i - 1):
                P[j, i + 1] = ((Xdata_comp[k] - x_abs[j]) * P[j + 1, i] + (x_abs[j + i + 1] - Xdata_comp[k]) * P[
                    j, i]) / (
                                      x_abs[j + i + 1] - x_abs[j])

        y_inter[k] = P[0, n - 1]

        D = np.zeros((n, n))
        D[:, 0] = y_abs

        for i in range(n - 1):
            D[i, 1] = (D[i + 1, 0] - D[i, 0]) / (x_abs[i + 1] - x_abs[i])

        for i in range(1, n - 1):
            for j in range(n - i - 1):
                D[j, i + 1] = (P[j + 1, i] + (Xdata_comp[k] - x_abs[j]) * D[j + 1, i] - P[j, i] + (
                        x_abs[j + i + 1] - Xdata_comp[k]) * D[j, i]) / (x_abs[j + i + 1] - x_abs[j])

        yveli_inter[k] = D[0, n - 1]

    Z_inter = y_inter[0][0]
    VZ_inter = yveli_inter[0][0]

    # clock estimation

    y_absp = []
    for i in range(satellite_positions.nb_epochs):
        y_absp.append(satellite_positions.epochs[i][id_sat][3])

    if time_interval_ind + sat_csckinter_nump > len(Precise_time) - 1:
        x_abs = Precise_time[time_interval_ind - sat_csckinter_nump + 1:time_interval_ind + 2]
        y_abs = y_absp[time_interval_ind - sat_csckinter_nump + 1:time_interval_ind + 2]

    else:
        if not time_interval_ind:
            x_abs = Precise_time[time_interval_ind + 1:time_interval_ind + sat_csckinter_nump + 2]
            y_abs = y_absp[time_interval_ind + 1:time_interval_ind + sat_csckinter_nump + 2]
        else:
            x_abs = Precise_time[time_interval_ind:time_interval_ind + sat_csckinter_nump + 1]
            y_abs = y_absp[time_interval_ind:time_interval_ind + sat_csckinter_nump + 1]

    y_inter = np.zeros((len(Xdata_comp), 1))
    yveli_inter = np.zeros((len(Xdata_comp), 1))

    n = len(x_abs)

    for k in range(len(Xdata_comp)):
        xd = np.zeros(n)
        for i in range(n):
            xd[i] = abs(x_abs[i] - Xdata_comp[k])
        i = sorted(range(len(xd)), key=lambda m: xd[m])

        x_abs_sorted = []
        y_abs_sorted = []
        for ii in range(len(i)):
            x_abs_sorted.append(x_abs[i[ii]])
            y_abs_sorted.append(y_abs[i[ii]])

        P = np.zeros((n, n))
        P[:, 0] = y_abs

        for i in range(n - 1):
            for j in range(n - i - 1):
                P[j, i + 1] = ((Xdata_comp[k] - x_abs[j]) * P[j + 1, i] + (x_abs[j + i + 1] - Xdata_comp[k]) * P[
                    j, i]) / (
                                      x_abs[j + i + 1] - x_abs[j])

        y_inter[k] = P[0, n - 1]

        D = np.zeros((n, n))
        D[:, 0] = y_abs

        for i in range(n - 1):
            D[i, 1] = (D[i + 1, 0] - D[i, 0]) / (x_abs[i + 1] - x_abs[i])

        for i in range(1, n - 1):
            for j in range(n - i - 1):
                D[j, i + 1] = (P[j + 1, i] + (Xdata_comp[k] - x_abs[j]) * D[j + 1, i] - P[j, i] + (
                        x_abs[j + i + 1] - Xdata_comp[k]) * D[j, i]) / (x_abs[j + i + 1] - x_abs[j])

        yveli_inter[k] = D[0, n - 1]

    clock_inter = y_inter[0][0]
    clock_drift_inter = yveli_inter[0][0]

    Rsat = [X_inter, Y_inter, Z_inter]
    Vsat = [VX_inter, VY_inter, VZ_inter]

    relativity = (Rsat * np.transpose(Vsat))/ C**2
    clock_error = clock_inter -2 * C * relativity
    satclock_vect = [clock_error,  clock_drift_inter]
    Rsat.append(clock_inter)

    return Rsat, Vsat, satclock_vect


def lagrande_interpolation(t, v, T):
    """% This file intepolates a function defined by a set of abscissa and a
    % corresponding set of values. The interpolation is done at given points
    % thanks to the Lagrande method and the interpolated value is returned.

    %   INPUTS :
    %       t : a vector containing the abscissa points at which the function
    %           is sampled ; t = [x1 x2 ... x_n]
    %       v : a vector containing the value of the function at the points
    %           given by t ; v = [f(x1) f(x2) ... f(x_n)]
    %       T : a vector or a value of the point(s) where to interpolate the
    %           value."""
    n = len(t)
    sum = 0
    for i in range(n):
        prod = v[i]
        for j in range(n):
            if i != j:
                prod = prod * (T - t[j]) / (t[i] - t[j])
        sum += prod
    return sum


def check_interpolation_accuracy(accurate_pos_filepath, interpolated_data):
    """This function was designed to verify the interpolation consistency. It is now of no use."""
    nb_samples = len(interpolated_data[:, 0])
    accurate_data = np.zeros((1800, 3))
    time = np.linspace(1, nb_samples, nb_samples)
    with open(accurate_pos_filepath) as sat_accurate_pos:
        nb_line = 0
        for line in sat_accurate_pos:
            data = line.strip("\n").split(',')
            accurate_data[nb_line, :] = [float(data[0]), float(data[1]), float(data[2])]
            nb_line += 1
    print(accurate_data[0, :])
    plt.plot(time[:nb_samples], accurate_data[:nb_samples, 0], "-k", linewidth=2, label="ref")
    plt.plot(time[:nb_samples], interpolated_data[:, 0], "-r", linewidth=2, label="interp")
    plt.title("x coordinate over time")
    plt.xlabel("time (s)")
    plt.ylabel("x (m)")
    plt.legend()
    plt.show()
    plt.plot(time[:nb_samples], accurate_data[:nb_samples, 0] - interpolated_data[:, 0], "-g", linewidth=2)
    plt.title("x error over time")
    plt.xlabel("time (s)")
    plt.ylabel("x (m)")
    plt.show()


def sat_position_from_Keplerian_elements(keplerian_elements, epoch):
    """this function computes the ECEF position of the satellites at epoch 'epoch', given the Keplerian parameters
    in the almanach.
    INPUTS :
        keplerian_elements = a dictionary with the sat IDs as keys and the Keplerian elements as values (in ISU)
        epoch : epoch at which to compute the position of the satellites, in GPS time (seconds after 01/06/1980)
    OUTPUT :
        keplerian_elements : the same dictionary as given in input, with an extra field 'pos' giving the ECEF position
        of the satellite as a 3x1 array (in meter)"""

    for sat in keplerian_elements:

        true_anomaly_offset = (epoch % 43082) / 43082 * 2 * np.pi
        keplerian_elements[sat]["pos"] = np.transpose(conversion.Keplerian2ECEF(keplerian_elements[sat]["a"],
                                            keplerian_elements[sat]["e"], keplerian_elements[sat]["i"],
                                            keplerian_elements[sat]["Omega"], keplerian_elements[sat]["omega"],
                                            keplerian_elements[sat]["true_anomaly"] + true_anomaly_offset, epoch))[0]

    return keplerian_elements
