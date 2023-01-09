######################################
###    AUTHOR : MATHIEU HUSSONG    ###
###   ENAC TELECOM LAB COPYRIGHT   ###
###        DATE : 07/12/2021       ###
###          VERSION : 1           ###
######################################

# THIS CODE COMPUTES THE PSEUDODISTANCES THAT SHOULD BE OBTAINED BY A GNSS RECEIVER ALONG A FLIGHT PROFILE


#######################
### LIBRARY IMPORTS ###
#######################
import numpy as np
import numpy.random as rdm
import sys
from sys import platform

if platform == "linux" or platform == "linux2":
    OPERATING_SYSTEM = "linux"
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.colors as colors
elif platform == "darwin":
    OPERATING_SYSTEM = "mac"
elif platform == "win32":
    OPERATING_SYSTEM = "windows"
else:
    OPERATING_SYSTEM = "not found"

if OPERATING_SYSTEM == "windows":
    ABSOLUTE_PATH = "C:/Users/Mathieu/Documents/GENE_COUPLE/"
else:
    ABSOLUTE_PATH = '/home/mathieu/GENE_COUPLE/'
sys.path.append(ABSOLUTE_PATH + 'GENERAL')
sys.path.append(ABSOLUTE_PATH + 'TRAJECTORY_GENERATION')
sys.path.append(ABSOLUTE_PATH + 'GNSS_GENERATION')
sys.path.append(ABSOLUTE_PATH + 'GNSS_STANDALONE_POSITIONING')
sys.path.append(ABSOLUTE_PATH + 'MEACONER_PROJECT')

if platform == "win32":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.colors as colors
#import waypoints
from decimal import *
import conversion
from constants import *
import sat_position_interpolation
#import meaconer_bias_estimation

C           = 299792458
ASTRO_UNIT  = 6.957e8
OMEGA_EARTH = 7.2921159e-5
EARTH_RADIUS= 6378137.0 # WGS-84 Earth Equatorial Radius
DEG2RAD     = np.pi / 180
RAD2DEG     = 180 / np.pi  # to convert from radians to degrees
WGS84_A     = 6378137  # the semi-major axis of the WGS84 model
WGS84_B     = 6356752.31424518  # the semi-minor axis of the WGS84 model
WGS84_F     = 1 / 298.257223563  # the flatness of the WGS84 model, we also have WGS84_F = (WGS84_A - WGS84_B) / WGS84_A
WGS84_E     = np.sqrt((WGS84_A**2 - WGS84_B**2) / WGS84_A**2)  # the first eccentricity of the WGS84 model
WGS84_E_PRIME = np.sqrt((WGS84_A**2 - WGS84_B**2) / WGS84_B**2)  # the second eccentricity of the WGS84 model
###############################
### USER-DEFINED PARAMETERS ###
###############################

getcontext().prec = 21 # increases the numerical precision of float computations when needed

# In this section and for boolean choices, the convention
# "0 = False = disabled = off " and "1 = True = enabled = on" is chosen

TRACK = "test.fp"  # Flight profile to generate pseudodistances from
SAVE_TRACK = "test.msr"
SP3_FILEPATH = "SATELLITE_POSITION_FILES/WUM0MGXULA_20201010000_01D_05M_ORB.SP3"  # SP3 filepath of the simulation epoch
GPS_ANTEX_FILEPATH = "ANTEX_FILES/gpsantex2020.txt"  # GPS ANTEX filepath of the simulation epoch
GAL_ANTEX_FILEPATH = "ANTEX_FILES/galileoantex2020.txt"  # Galileo ANTEX filepath of the simulation epoch
GLO_ANTEX_FILEPATH = "ANTEX_FILES/glonassantex2020.txt"  # Glonass ANTEX filepath of the simulation epoch
BDS_ANTEX_FILEPATH = "ANTEX_FILES/beidouantex2020.txt"  # Beidou ANTEX filepath of the simulation epoch

GPS = 1  # enables the pseudodistance generation with GPS satellites
GAL = 0  # enables the pseudodistance generation with Galileo satellites
GLO = 0  # enables the pseudodistance generation with Glonass satellites
BDS = 0  # enables the pseudodistance generation with Beidou satellites

L1 = 1  # enables the pseudodistance generation for L1 signals (only if "GPS" in enabled)
L2 = 0  # enables the pseudodistance generation for L2 signals (only if "GPS" in enabled)
L5 = 0  # enables the pseudodistance generation for L5 signals (only if "GPS" in enabled)
E1 = 0  # enables the pseudodistance generation for E1 signals (only if "GAL" in enabled)
E5 = 0  # enables the pseudodistance generation for E5 signals (only if "GAL" in enabled)
R1 = 0  # enables the pseudodistance generation for R1 signals (only if "GLO" in enabled)
R2 = 0  # enables the pseudodistance generation for R2 signals (only if "GLO" in enabled)
B2 = 0  # enables the pseudodistance generation for B2 signals (only if "BDS" in enabled)
B3 = 0  # enables the pseudodistance generation for B3 signals (only if "BDS" in enabled)

ELEVATION_MASK = 2  # minimum required elevation angle to generate the pseudodistances (in degrees)

GENERATED_SAMPLING_FREQUENCY = 1  # (Hz) desired sampling frequency of the pseudodistance generation
# (must be a divider of the track sampling frequency)
# set "GENERATED_SAMPLING_FREQUENCY" to 0 if you want to keep the same sampling frequency for the pseudodistance
# generation than the one in the track

FREQUENCY_CORRELATION = 0  # cross-correlation (between 0 and 1) of the same kind errors between different frequencies

# All the following error terms can be enabled or disabled.
# If enabled, the error parameters refer to the induced variations of the pseudodistances caused by the error

SATELLITE_HARDWARE_BIAS = 0  # includes the satellite hardware bias errors in the pseudodistance generation
GPS_SATELLITE_HARDWARE_INITIAL_STD = 1  # initial standard deviation of the GPS hardware code biases (in meters)
GAL_SATELLITE_HARDWARE_INITIAL_STD = 0.5  # initial standard deviation of the Galileo hardware code biases (in meters)
GLO_SATELLITE_HARDWARE_INITIAL_STD = 0.5  # initial standard deviation of the Glonass hardware code biases (in meters)
BDS_SATELLITE_HARDWARE_INITIAL_STD = 1  # initial standard deviation of the Beidou hardware code biases (in meters)

RECEIVER_HARDWARE_BIAS = 0  # includes the receiver hardware bias errors in the pseudodistance generation
GPS_RECEIVER_HARDWARE_INITIAL_STD = 0.3  # initial standard deviation of the GPS hardware code biases (in meters)
GAL_RECEIVER_HARDWARE_INITIAL_STD = 0.1  # initial standard deviation of the Galileo hardware code biases (in meters)
GLO_RECEIVER_HARDWARE_INITIAL_STD = 0.1  # initial standard deviation of the Glonass hardware code biases (in meters)
BDS_RECEIVER_HARDWARE_INITIAL_STD = 0.5  # initial standard deviation of the Beidou hardware code biases (in meters)

RECEIVER_CLOCK_BIAS = 0  # includes the receiver clock bias errors in the pseudodistance generation
RECEIVER_CLOCK_INITIAL_STD = 3e5  # initial standard deviation of the receiver clock offset (in meters)
RECEIVER_CLOCK_CORRELATION_TIME = 1e8  # defines the correlation time of the Rx clock random walk error (in seconds)
RECEIVER_CLOCK_RANDOM_WALK_PARAMETER = 2e5  # defines the parameter of the Rx clock random walk error (in m/Hz^0.5)

EPHEMERIS_ERROR = 1  # includes the ephemeris errors in the pseudodistance generation
EPHEMERIS_CORRELATION_TIME = 1500  # defines the correlation time of the ephemeris random walk error (in seconds)
EPHEMERIS_RANDOM_WALK_PARAMETER = 0.1  # defines the parameter of the ephemeris random walk error (in m/Hz^0.5)

IONOSPHERIC_ERROR = 1  # includes the ionospheric errors in the pseudodistance generation
VERTICAL_TOTAL_ELECTRON_CONTENT_UNIT = 20  # defines the vertical TECU in the ionosphere during the whole simulation
IONO_HEIGHT = 500e3  # defines the ionosphere characteristic height (in meters)

TROPOSPHERIC_ERROR = 1  # includes the tropospheric errors in the pseudodistance generation
TROPOSPHERIC_INITIAL_STD = 0.1  # initial standard deviation of the vertical wet tropospheric error (in meters)
TROPOSPHERIC_CORRELATION_TIME = 36000  # defines the correlation time of the tropospheric random walk error (in seconds)
TROPOSPHERIC_RANDOM_WALK_PARAMETER = 0.0002  # defines the parameter of the tropospheric random walk error (in m/Hz^0.5)

MULTIPATH_ERROR = 1  # includes the multipath errors in the pseudodistance generation
MULTIPATH_CORRELATION_DISTANCE = 20  # defines the correlation distance of the multipath random walk error (in meters)
MULTIPATH_RANDOM_WALK_PARAMETER = 0.5  # defines the parameter of the code multipath random walk error (in m/Hz^0.5)

THERMAL_NOISE = 1  # includes the thermal noise errors in the pseudodistance generation
THERMAL_NOISE_CODE_STD = 0.3  # defines the standard deviation of the thermal noise code error (in meters)
THERMAL_NOISE_PHASE_STD = 0.01  # defines the standard deviation of the thermal noise phase error (in meters)

CARRIER_WIND_UP = 1  # includes the carrier wind-up effect in the carrier phase pseudodistances

CARRIER_INTEGER_AMBIGUITY = 1  # includes a random carrier integer ambiguity error in the phase pseudodistance

SAGNAC_EFFECT = 0  # includes the special relativity Sagnac error term in the pseudodistances

SHAPIRO_EFFECT = 0  # includes the general relativity Shapiro error term in the pseudodistances

DISPLAY_RESULTS = 1 # plots the generates results

CODE_FILTERING_BY_THE_PHASE = 1
PHASE_FILTER_ORDER = 100


###############
### CLASSES ###
###############

class SatPosition:
    """SatPosition class contains the data from the SP3 files describing
    the ECEF location of the satellites at given epochs"""
    def __init__(self):
        self.epoch_start = ""
        self.epoch_end = ""
        self.epochs = []
        self.nb_epochs = 0
        self.nb_pos = 0
        self.label = "unnamed"

    def __repr__(self):
        return "The SatPosition class '{}' contains {} satellite positions over {} epochs from the {}.".format(
            self.label, self.nb_pos, self.nb_epochs, self.epoch_start)


class Pseudodistances:
    """Pseudodistances class contains the pseudoranges of the satellites at given epochs"""
    def __init__(self):
        self.epoch_start = ""
        self.epochs = []
        self.nb_epochs = 0
        self.nb_pseudoranges = 0
        self.frequencies = {"GPS": [], "GAL": [], "GLO": [], "BDS": []}
        self.label = "unnamed"

    def __repr__(self):
        return "The Pseudodistances class '{}' contains {} satellite pseudodistances over {} epochs from the {}.".format(
            self.label, self.nb_pseudoranges, self.nb_epochs, self.epoch_start)

    def save(self, filename):
        """this function saves the generated pseudodistances in a .txt file
        INPUT : filename = the name of the .txt file that is stored in the '/GENERATED_MEASUREMENTS' folder"""
        file_location = "GENERATED_MEASUREMENTS/" + filename
        file = open(file_location, "w")
        file.writelines("# Generated pseudodistances file label :\n{}\n".format(self.label))
        file.writelines("# Starting date of the measurements :\n{}\n".format(self.epoch_start[0:9]))
        file.writelines("# Number of epochs in the measurements :\n{}\n".format(self.nb_epochs))
        file.writelines(
            "# Number of pseudodistances in the measurements :\n{}\t({} code and {} phase pseudodistances)\n\n".format(
            self.nb_pseudoranges*2, self.nb_pseudoranges, self.nb_pseudoranges))
        file.writelines("Time (GPS)\t\tSat\t\tFrequency\tCode PSR (m)\t\tPhase PSR (m)\n")
        text = []
        for i in range(self.nb_epochs):
            time = str(int(100*self.epochs[i]["time"])/100)
            for sat, data_sat in self.epochs[i].items():
                if sat != "time":
                    for freq, pseudos in data_sat.items():
                        code_psr = str(int(100000*pseudos["code"])/100000)
                        phase_psr = str(int(100000*pseudos["phase"])/100000)
                        text.append(time + "\t\t" + sat + "\t\t" + freq + "\t\t" + code_psr + "\t\t" + phase_psr + "\n")
        file.writelines(text)
        file.close()

    def save_in_msr(self, filename):
        """this function saves the generated pseudodistances in a file in the RINEX MSR standard
        INPUT : filename = the name of the .rnx file that is stored in the '/GENERATED_MEASUREMENTS' folder"""
        file_location = "GENERATED_MEASUREMENTS/" + filename
        file = open(file_location, "w")
        text = []
        for i in range(self.nb_epochs):
            time = str(int(100*self.epochs[i]["time"])/100)
            for sat, data_sat in self.epochs[i].items():
                if sat != "time":
                    for freq, pseudos in data_sat.items():
                        code_psr = str(int(100000*pseudos["code"])/100000)
                        phase_psr = str(int(100000*pseudos["phase"] / C * find_frequency_from_band(freq))/100000)
                        doppler = str(int(1000*pseudos["doppler"] / C * find_frequency_from_band(freq))/1000)
                        cn0 = str(int(100*pseudos["CN0"] / C * find_frequency_from_band(freq))/100)
                        rinex_freq = sat[0] + freq[1] + "X"
                        text.append(time + "\tGENE_COUPLE\t" + rinex_freq + "\t" + str(int(sat[1:]))
                                    + "\t" + code_psr + "\t" + phase_psr + "\t" + doppler + "\t" + cn0 + "\n")
        file.writelines(text)
        file.close()


def load_pseudodistances_from_msr(filename):
    """this function loads and returns a Pseudodistance object from a .msr file"""
    pseudos = Pseudodistances()
    file = open("../GNSS_GENERATION/GENERATED_MEASUREMENTS/" + filename, "r")
    last_time = -1
    gps_freq =[]
    gal_freq = []
    glo_freq = []
    bds_freq = []

    for line in file:
        data = line.strip("\n").split()
        time = float(data[0])
        if np.abs(time - last_time) > EPSILON:
            if last_time > 0:
                pseudos.epochs.append(dico)
                pseudos.nb_epochs += 1
                pseudos.nb_pseudoranges += nb_pseudos
            else:
                pseudos.epoch_start = time
            last_time = time
            dico = {"time": time}
            nb_pseudos = 0
        freq = data[2]
        prn = int(data[3])
        if prn < 10:
            sat = freq[0] + "0" + str(prn)
        else:
            sat = freq[0] + str(prn)
        band = freq_from_msr(freq)
        code = float(data[4])
        phase = float(data[5])
        doppler = float(data[6])
        cn0 = float(data[7])
        dico.setdefault(sat, {})
        dico[sat].setdefault(band, {"code": code, "phase": phase, "doppler": doppler, "cn0": cn0})
        if freq[0] == "G" and band not in gps_freq:
            gps_freq.append(band)
        elif freq[0] == "E" and band not in gal_freq:
            gal_freq.append(band)
        elif freq[0] == "R" and band not in glo_freq:
            glo_freq.append(band)
        elif freq[0] == "C" and band not in bds_freq:
            bds_freq.append(band)
        nb_pseudos += 2

    pseudos.epochs.append(dico)
    pseudos.frequencies["GPS"] = gps_freq
    pseudos.frequencies["GAL"] = gal_freq
    pseudos.frequencies["GLO"] = glo_freq
    pseudos.frequencies["BDS"] = bds_freq
    return pseudos


#################
### FUNCTIONS ###
#################


def load_keplerian_elements(keplerian_file):
    """this function reads the keplerian elements of the satellite constellation, and stores them in a dictionary
    structure with the IDs of the satellites as keys, and the Keplerian elements as values (in ISU)"""

    file = open(keplerian_file, "r")
    keplerian_elements = {}

    for line in file:
        data = line.strip("\n").split()
        if data:
            if data[0][0] == '*':
                prn = int(data[5][4:6])
                if prn < 10:
                    num = "G0" + str(prn)
                else:
                    num = "G" + str(prn)
                keplerian_elements[num] = {}
            elif data[0] == "ID:":
                keplerian_elements[num]["ID"] = int(data[1])
            elif data[0] == "Health:":
                keplerian_elements[num]["health"] = int(data[1])
            elif data[0] == "Eccentricity:":
                keplerian_elements[num]["e"] = float(data[1])
            elif data[0] == "Time":
                keplerian_elements[num]["time_of_applicability"] = float(data[3])
            elif data[0] == "Orbital":
                keplerian_elements[num]["i"] = float(data[2])
            elif data[0] == "Rate":
                keplerian_elements[num]["rate_of_right_ascen"] = float(data[4])
            elif data[0] == "SQRT(A)":
                keplerian_elements[num]["a"] = float(data[3]) ** 2
            elif data[0] == "Right":
                keplerian_elements[num]["Omega"] = float(data[4])
            elif data[0] == "Argument":
                keplerian_elements[num]["omega"] = float(data[3])
            elif data[0] == "Mean":
                keplerian_elements[num]["true_anomaly"] = float(data[2])
            elif data[0] == "Week:":
                keplerian_elements[num]["week"] = int(data[1])

    return keplerian_elements



def freq_from_msr(msr_freq):
    """this function returns the frequency name corresponding to the frequency abbreviation of the .msr file"""
    constellation = msr_freq[0]
    band = msr_freq[1]
    if constellation == "G":
        if band == "1":
            return "L1"
        elif band == "2":
            return "L2"
        else:
            return "L5"
    else:
        return constellation + band


def save_pos(sat, freq):
    """a function to be deleted after test validation"""
    if sat[0] == "G":
        if freq == "L1":
            return int(sat[1:]) - 1
        else:
            return int(sat[1:]) + 35
    elif sat[0] == "E":
        if freq == "E1":
            return int(sat[1:]) + 71
        else:
            return int(sat[1:]) + 107
    elif sat[0] == "R":
        if freq == "R1":
            return int(sat[1:]) + 143
        else:
            return int(sat[1:]) + 179
    elif sat[0] == "C":
        if freq == "B2":
            return int(sat[1:]) + 215
        else:
            return int(sat[1:]) + 251
    else:
        return -1


def save_sat(index):
    """a function to be deleted after test validation"""
    const = index // 72
    if const == 0:
        con = "G"
    elif const == 1:
        con = "E"
    elif const == 2:
        con = "R"
    else:
        con = "C"
    prn = index % 36 + 1
    if prn < 10:
        num = "0" + str(prn)
    else:
        num = str(prn)

    if 0 <= index <= 35:
        freq = "L1"
    elif 36 <= index <= 71:
        freq = "L5"
    elif 72 <= index <= 107:
        freq = "E1"
    elif 108 <= index <= 143:
        freq = "E5"
    elif 144 <= index <= 179:
        freq = "R1"
    elif 180 <= index <= 215:
        freq = "R2"
    elif 216 <= index <= 251:
        freq = "B2"
    else:
        freq = "B3"
    return con + num + " " + freq


def initial(constellation):
    if constellation == "GPS":
        return 'G'
    elif constellation == "GAL":
        return 'E'
    elif constellation == "GLO":
        return 'R'
    else:
        return 'C'


def constellation_name(initial):
    if initial == 'G':
        return "GPS"
    elif initial == 'E':
        return "GAL"
    elif initial ==  'R':
        return "GLO"
    else:
        return "BDS"


def sun_position(gps_time, leap_seconds=18):
    """This function computes the Sun position in the ECEF frame, along with the
    distance between the Sun and the Earth.

    %   INPUTS :
    %       the date in the gregorian style at which to compute the sun position
    %       the number of leap seconds at the epoch considered (by default :18)

    %   OUTPUTS :
    %       sun_position : the position of the Sun given in the ECEF-frame"""
    time_univ = gps_time - leap_seconds
    reference_time = conversion.gregorian2GPStime(2000, 1, 1, 12, 0, 0)[0]
    time_sun = (gps_time - reference_time) / 86400 / 365.25
    eps = 23.439291 - 0.0130042 * time_sun
    sine = np.sin(eps * DEG2RAD)
    cose = np.cos(eps * DEG2RAD)
    ms = 357.5277233 + 35999.05034 * time_sun
    ls = 280.460 + 36000.770 * time_sun + 1.914666471 * np.sin(ms * DEG2RAD) + 0.019994643 * np.sin(2.0 * ms * DEG2RAD)
    rs = ASTRO_UNIT * (1.000140612 - 0.016708617 * np.cos(ms * DEG2RAD) - 0.000139589 * np.cos(2.0 * ms * DEG2RAD))
    sinl = np.sin(ls * DEG2RAD)
    cosl = np.cos(ls * DEG2RAD)
    sun_posx = rs * cosl
    sun_posy = rs * cose * sinl
    sun_posz = rs * sine * sinl
    sun_posECI = [sun_posx, sun_posy, sun_posz]
    sun_position = conversion.ECI2ECEF(sun_posECI, time_univ)
    return sun_position


def wind_up_error(user_position, sat_position, epoch, frequency, last_wind_up=0):
    """this function computes the phase wind-up error term of a signal
    INPUTS :
        user_position : the 3x1 ECEF position of the receiver (in meters)
        sat_position : te 3x1 ECEF position of the satellite (in meters)
        epoch : the epoch (in GPS time) at which to compute the wind up error
        frequency : the frequency of the signal in use (in Hertz)
        last_wind_up : wind-up term computed at the previous epoch for the same signal (in meters)
    OUTPUTS :
        wind_up_error : the error (in meters) induced by the wind-up effect
                        a positive error increases the received pseudodistance"""
    phi, lamda, alt = conversion.ECEF2LLA(user_position)

    # rotation_matrix transforms ENU 2 ECEF frames
    rotation_matrix = np.linalg.inv(np.array([[-np.sin(lamda), np.cos(lamda), 0],
                                [-np.sin(phi)*np.cos(lamda), -np.sin(phi)*np.sin(lamda), np.cos(phi)],
                                [np.cos(phi)*np.cos(lamda), np.cos(phi)*np.sin(lamda), np.sin(phi)]]))

    los_vector = - np.transpose(np.array([user_position[0] - sat_position[0],
                           user_position[1] - sat_position[1],
                           user_position[2] - sat_position[2]]))
    los_vector_norm = np.linalg.norm(los_vector)
    los_vector /= los_vector_norm
    com2ecef = conversion.CoM2ECEF(sun_position(epoch), sat_position)
    delta_s = com2ecef[:, 0] - los_vector * np.dot(np.transpose(los_vector), com2ecef[:, 0]) \
              - np.cross(los_vector, com2ecef[:, 1])
    delta_r = rotation_matrix[:, 0] - los_vector * np.dot(np.transpose(los_vector), rotation_matrix[:, 0]) \
              + np.cross(los_vector, rotation_matrix[:, 1])

    delta_phase_wind_up = np.sign(np.dot(los_vector, np.cross(delta_s, delta_r))) * np.arccos(
        np.dot(delta_s, delta_r) / np.linalg.norm(delta_s) / np.linalg.norm(delta_r))

    wind_up_error = C / frequency * delta_phase_wind_up / 2 / np.pi
    if wind_up_error - last_wind_up > C / frequency / 2:
        while wind_up_error - last_wind_up > C / frequency / 2:
            wind_up_error -= C / frequency
    if wind_up_error - last_wind_up < -C / frequency / 2:
        while wind_up_error - last_wind_up < -C / frequency / 2:
            wind_up_error += C / frequency
    return wind_up_error


def sagnac_error(satellite_position, satellite_velocity):
    """this function computes the Sagnac special relativity induced error.
    INPUTS:
        user_position : the 3x1 ECEF position of the receiver (in meters)
        sat_position : te 3x1 ECEF position of the satellite (in meters)
        relative_velocity : the 3x1 ECEF relative velocity of the user with respect to the satellite (in m/s)
    OUTPUTS:
        sagnac : the sagnac error term in the pseudodistance error budget (in meters)
            a positive sagnac error increases the received pseudodistance"""
    sagnac = - 2 * np.dot(satellite_position, np.transpose(satellite_velocity)) / C
    return sagnac


def shapiro_error(user_position, satellite_position):
    """this function computes the Shapiro general relativity induced error.
    INPUTS:
        user_position : the 3x1 ECEF position of the receiver (in meters)
        sat_position : te 3x1 ECEF position of the satellite (in meters)
    OUTPUTS:
        shapiro : the Shapiro error term in the pseudodistance error budget (in meters)
            a positive Shapiro error increases the received pseudodistance"""
    norm_user_pos = np.linalg.norm(user_position)
    norm_sat_pos = np.linalg.norm(satellite_position)
    norm_user_sat_norm = np.linalg.norm(satellite_position - user_position)
    shapiro = 2 * MU / C**2 * np.log((norm_sat_pos + norm_user_pos + norm_user_sat_norm) /
                                     (norm_sat_pos + norm_user_pos - norm_user_sat_norm))
    return shapiro


def ionospheric_error(frequency, elevation):
    """this function computes the ionospheric error.
        INPUTS:
            user_position : the 3x1 ECEF position of the receiver (in meters)
            sat_position : te 3x1 ECEF position of the satellite (in meters)
            frequency : the frequency of the signal in use (in Hertz)
            elevation : the elevation of the satellite as seen by the user (in radians)
        OUTPUTS:
            iono_error : the ionospheric error term in the pseudodistance error budget (in meters)
                a positive ionospheric error increases the received code pseudodistance
                and decreases the received phase pseudodistance"""
    obliquity_factor = 1 / np.cos(np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + IONO_HEIGHT) * np.cos(elevation)))
    slant_total_electron_content = VERTICAL_TOTAL_ELECTRON_CONTENT_UNIT * obliquity_factor
    iono_coefficient = 40.3 * 10**16 / frequency ** 2
    iono_error = iono_coefficient * slant_total_electron_content
    return iono_error


def tropospheric_error(user_position, elevation, delta_tropo_zenith_wet, day_of_year=128):
    """this function computes the ionospheric error.
    INPUTS:
        user_position : the 3x1 ECEF position of the receiver (in meters)
        elevation : the elevation of the satellite as seen by the user (in radians)
        delta_tropo_zenith_wet : the random part of the tropospheric vertical wet delay (in meters)
    OUTPUTS:
        tropo_error : the tropospheric error term in the pseudodistance error budget (in meters)
            a positive tropospheric error increases the received pseudodistances"""
    # determination of the coefficients of the Niel mapping functions
    latitude, _, altitude = conversion.ECEF2LLA(user_position)
    a_d_avg = np.interp(latitude * RAD2DEG, [15, 30, 45, 60, 75],
                    [1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3])
    b_d_avg = np.interp(latitude * RAD2DEG, [15, 30, 45, 60, 75],
                    [2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3])
    c_d_avg = np.interp(latitude * RAD2DEG, [15, 30, 45, 60, 75],
                    [62.610505e-3, 62.837191e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3])
    a_d_amp = np.interp(latitude * RAD2DEG, [15, 30, 45, 60, 75],
                        [0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5])
    b_d_amp = np.interp(latitude * RAD2DEG, [15, 30, 45, 60, 75],
                        [0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5])
    c_d_amp = np.interp(latitude * RAD2DEG, [15, 30, 45, 60, 75],
                        [0, 9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5])

    a_d = a_d_avg - a_d_amp * np.cos(2 * np.pi * (day_of_year - 28) / 365.25)
    b_d = b_d_avg - b_d_amp * np.cos(2 * np.pi * (day_of_year - 28) / 365.25)
    c_d = c_d_avg - c_d_amp * np.cos(2 * np.pi * (day_of_year - 28) / 365.25)

    a_h = 2.53e-5
    b_h = 5.49e-3
    c_h = 1.14e-3

    a_w = np.interp(latitude * RAD2DEG, [15, 30, 45, 60, 75],
                    [5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4])
    b_w = np.interp(latitude * RAD2DEG, [15, 30, 45, 60, 75],
                    [1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3])
    c_w = np.interp(latitude * RAD2DEG, [15, 30, 45, 60, 75],
                    [4.3472961e-2, 4.6729510e-2, 4.3908930e-2, 4.4626982e-2, 5.4736038e-2])

    mapping_dry = (1 + a_d / (1 + b_d / (1 + c_d))) / \
                  (np.sin(elevation) + a_d / (np.sin(elevation) + b_d / (np.sin(elevation) + c_d))) + altitude / 1e3 * \
                  (1 / np.sin(elevation) - (1 + a_h / (1 + b_h / (1 + c_h))) / (np.sin(elevation) +
                  a_h / (np.sin(elevation) + b_h/ (np.sin(elevation) + c_h))))
    mapping_wet = (1 + a_w / (1 + b_w / (1 + c_w))) / \
                  (np.sin(elevation) + a_w / (np.sin(elevation) + b_w / (np.sin(elevation) + c_w)))
    tropo_zen_dry = 2.3 * np.exp(-0.116e-3 * altitude / 1e3)
    tropo_zen_wet = 0.1 + delta_tropo_zenith_wet
    tropo_error = tropo_zen_dry * mapping_dry + tropo_zen_wet * mapping_wet
    return tropo_error


def maximum_antenna_gain(elevation):
    """this function returns the maximum antenna gain of the aircraft GNSS antenna, based on DO235C."""
    elevation_degree = elevation * RAD2DEG
    if elevation_degree < -30:
        return 10 ** (-10 / 10)
    if elevation_degree <= 0:
        return 10 ** ((elevation_degree - 30) / 6 / 10)
    if elevation_degree < 10:
        return 10 ** ((-2 + 0.45 * elevation_degree) / 10)
    if elevation_degree < 75:
        return 10 ** ((2.5 + (elevation_degree - 10) / 130) / 10)
    else:
        return 10 ** (3 / 10)


def minimum_antenna_gain(elevation):
    """this function returns the maximum antenna gain of the aircraft GNSS antenna, based on DO235C.
    for negative elevations, there is no data so an extrapolation is made. The results may therefore be distorted when
    using this function for negative elevations."""
    elevation_degree = elevation * RAD2DEG
    if elevation_degree < 20:
        return 10 ** ((-7 + 0.3 * elevation_degree) / 10)
    if elevation_degree < 75:
        return 10 ** ((-1 + 3 * (elevation_degree - 20) / 110) / 10)
    else:
        return 10 ** (0.5 / 10)


def distance(pos_a, pos_b):
    """retuns the euclidian distance between two ECEF points which are not numpy arrays"""
    return np.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2 + (pos_a[2] - pos_b[2])**2)


def read_satellite_position(sp3_filepath, gps_antex_filepath="", gal_antex_filepath="", glo_antex_filepath="",
                            bds_antex_filepath=""):
    """this functions returns the center of mass of each satellite (GPS+GAL+GLO+BDS) sampled each 30 seconds.
    The data are drawn from a SP3 file given as input.
    The center of mass can be corrected to be the center of phase of the frequency in use, with the input of an
    ANTEX file"""
    if GPS:
        gps_antex = np.zeros((32, 3))
    if GAL:
        gal_antex = np.zeros((36, 3))
    if GLO:
        glo_antex = np.zeros((27, 3))
    if BDS:
        bds_antex = np.zeros((59, 3))

    if GPS and gps_antex_filepath:
        with open(gps_antex_filepath) as antex:
            for line in antex:
                data = line.strip("\n").split()
                sat_id = int(data[0])
                x = float(data[1]) / 1000
                y = float(data[2]) / 1000
                z = float(data[3]) / 1000
                gps_antex[sat_id-1, :] = [x, y, z]
    if GAL and gal_antex_filepath:
        with open(gal_antex_filepath) as antex:
            for line in antex:
                data = line.strip("\n").split()
                sat_id = int(data[0])
                x = float(data[1]) / 1000
                y = float(data[2]) / 1000
                z = float(data[3]) / 1000
                gal_antex[sat_id-1, :] = [x, y, z]
    if GLO and glo_antex_filepath:
        with open(glo_antex_filepath) as antex:
            for line in antex:
                data = line.strip("\n").split()
                sat_id = int(data[0])
                x = float(data[1]) / 1000
                y = float(data[2]) / 1000
                z = float(data[3]) / 1000
                glo_antex[sat_id-1, :] = [x, y, z]
    if BDS and bds_antex_filepath:
        with open(bds_antex_filepath) as antex:
            for line in antex:
                data = line.strip("\n").split()
                sat_id = int(data[0])
                x = float(data[1]) / 1000
                y = float(data[2]) / 1000
                z = float(data[3]) / 1000
                bds_antex[sat_id-1, :] = [x, y, z]

    with open(sp3_filepath) as sp3:
        satellite_positions = SatPosition()
        satellite_positions.label = sp3_filepath[:-4]
        enabled_constellations = []
        if GPS:
            enabled_constellations.append("G")
        if GAL:
            enabled_constellations.append("E")
        if GLO:
            enabled_constellations.append("R")
        if BDS:
            enabled_constellations.append("C")
        deja_vu_epoch = False

        for line in sp3:
            data = line.strip("\n").split()

            if data[0][0] == "*":
                year = int(data[1])
                month = int(data[2])
                day = int(data[3])
                hour = int(data[4])
                minute = int(data[5])
                second = float(data[6])
                if not deja_vu_epoch:
                    satellite_positions.epoch_start = "{}/{}/{} at {}::{}::{}".format(day, month, year, hour, minute,
                                                                                    int(second))
                    deja_vu_epoch = True
                gps_time = round(conversion.gregorian2GPStime(year, month, day, hour, minute, second)[0])
                satellite_positions.epochs.append({"time": gps_time})

                satellite_positions.nb_epochs += 1

            if data[0][0] == "P" and data[0][1] in enabled_constellations:
                sat_id = data[0][1:4]
                x = float(data[1]) * 1000
                y = float(data[2]) * 1000
                z = float(data[3]) * 1000
                C = 299792458
                time = float(data[4]) * C * 1e-6
                satellite_positions.epochs[-1][sat_id] = [x, y, z, time]
                satellite_positions.nb_pos += 1

        satellite_positions.epoch_end = "{}/{}/{} at {}:{}:{}".format(day, month, year, hour, minute, int(second))

    # refinement of the satellite positions based on the ANTEX data
    for epoch in range(satellite_positions.nb_epochs):
        sun_pos_ecef = sun_position(satellite_positions.epochs[epoch]["time"])
        for sat, sat_position in satellite_positions.epochs[epoch].items():
            if sat[0] == "G":
                com2ecef = conversion.CoM2ECEF(sun_pos_ecef, sat_position[1:4])
                pco_sat = np.dot(com2ecef, gps_antex[int(sat[1:])-1, :])
                calibrated_sat_position = sat_position[1:4] + pco_sat
                satellite_positions.epochs[epoch][sat][1:4] = calibrated_sat_position
            if sat[0] == "E":
                com2ecef = conversion.CoM2ECEF(sun_pos_ecef, sat_position[1:4])
                pco_sat = np.dot(com2ecef, gal_antex[int(sat[1:])-1, :])
                calibrated_sat_position = sat_position[1:4] + pco_sat
                satellite_positions.epochs[epoch][sat][1:4] = calibrated_sat_position
            if sat[0] == "R":
                com2ecef = conversion.CoM2ECEF(sun_pos_ecef, sat_position[1:4])
                pco_sat = np.dot(com2ecef, glo_antex[int(sat[1:])-1, :])
                calibrated_sat_position = sat_position[1:4] + pco_sat
                satellite_positions.epochs[epoch][sat][1:4] = calibrated_sat_position
            if sat[0] == "C":
                com2ecef = conversion.CoM2ECEF(sun_pos_ecef, sat_position[1:4])
                pco_sat = np.dot(com2ecef, bds_antex[int(sat[1:])-1, :])
                calibrated_sat_position = sat_position[1:4] + pco_sat
                satellite_positions.epochs[epoch][sat][1:4] = calibrated_sat_position

    return satellite_positions


def euclidian_distance(user_position, satellite_positions, epoch, elevation_mask=0):
    """returns the Euclidian distances between the user and all the visible satellites"""

    satellite_positions_at_reception_epoch, satellite_velocities_at_reception_epoch = \
        sat_position_interpolation.neville_interpolation(satellite_positions, epoch)

    satellite_elevations_azimuths = {}
    for sat, sat_position in satellite_positions_at_reception_epoch.items():
        elevation, azimuth = conversion.ECEF2elevation_azimuth(user_position, sat_position[0:3])
        satellite_elevations_azimuths[sat] = [elevation, azimuth]

    euclidian_distances = {}
    for sat, sat_position in satellite_positions_at_reception_epoch.items():
        if satellite_elevations_azimuths[sat][0] * 180 / np.pi > elevation_mask:
            distance = np.sqrt((user_position[0] - sat_position[0])**2 + (user_position[1] - sat_position[1])**2
                               + (user_position[2] - sat_position[2])**2)
            euclidian_distances[sat] = {"distance": distance, "tx_time": epoch}

    # refinement of the distance and the emission time taking into account the displacement of the satellite in the
    # transmission duration :
    tolerable_error = 1/2**26 # delta_tx error = 1.5*10^-8 s ==> delta_pos error < 5*10^-5 m
    for sat, sat_dist_and_time in euclidian_distances.items():
        error = Decimal(epoch) - Decimal(sat_dist_and_time["tx_time"]) - Decimal(sat_dist_and_time["distance"]) / C
        tx_epoch = Decimal(sat_dist_and_time["tx_time"])
        sat_distance = sat_dist_and_time["distance"]
        while abs(error) > tolerable_error:
            tx_epoch += error
            pos_dict, vel_dict = sat_position_interpolation.neville_interpolation(satellite_positions,
                                tx_epoch, sat)
            new_sat_pos = pos_dict[sat]
            new_sat_vel = vel_dict[sat]
            sat_distance = np.sqrt((user_position[0] - new_sat_pos[0])**2 + (user_position[1] - new_sat_pos[1])**2
                               + (user_position[2] - new_sat_pos[2])**2)
            error = Decimal(epoch) - tx_epoch - Decimal(sat_distance) / C

        euclidian_distances[sat] = {"distance": sat_distance,
                            "tx_time": tx_epoch,
                            "pos": new_sat_pos[0:3], "vel": new_sat_vel, "clock": new_sat_pos[3]}

        # compensating the earth rotation at the actual emission time
        theta = OMEGA_EARTH * float((Decimal(epoch) - tx_epoch))
        rotation_theta = np.array([[np.cos(theta), np.sin(theta), 0],
                                    [-np.sin(theta), np.cos(theta), 0],
                                   [0, 0, 1]])
        corrected_sat_pos = np.transpose(np.dot(rotation_theta, [[new_sat_pos[0]], [new_sat_pos[1]], [new_sat_pos[2]]]))[0]
        corrected_sat_vel = np.transpose(np.dot(rotation_theta, [[new_sat_vel[0]], [new_sat_vel[1]], [new_sat_vel[2]]]))[0]

        corrected_distance = np.sqrt((user_position[0] - corrected_sat_pos[0])**2 + (user_position[1] - corrected_sat_pos[1])**2
                               + (user_position[2] - corrected_sat_pos[2])**2)
        euclidian_distances[sat] = {"distance": corrected_distance,
                                    "tx_time": tx_epoch,
                                    "pos": corrected_sat_pos, "vel": corrected_sat_vel, "clock": new_sat_pos[3]}

    return euclidian_distances


def euclidian_distance_from_almanach(user_position, satellite_positions, epoch, elevation_mask=0):
    """returns the Euclidian distances between the user and all the visible satellites"""

    satellite_positions_at_reception_epoch = sat_position_interpolation.sat_position_from_Keplerian_elements(
        satellite_positions, epoch)
    satellite_positions_at_previous_epoch = sat_position_interpolation.sat_position_from_Keplerian_elements(
        satellite_positions, epoch - 1)

    satellite_elevations_azimuths = {}
    for sat, sat_data in satellite_positions_at_reception_epoch.items():
        elevation, azimuth = conversion.ECEF2elevation_azimuth(user_position, sat_data["pos"])
        satellite_elevations_azimuths[sat] = [elevation, azimuth]

    euclidian_distances = {}
    for sat, sat_data in satellite_positions_at_reception_epoch.items():
        if satellite_elevations_azimuths[sat][0] * 180 / np.pi > elevation_mask:
            distance = np.sqrt((user_position[0] - sat_data["pos"][0])**2 + (user_position[1] - sat_data["pos"][1])**2
                               + (user_position[2] - sat_data["pos"][2])**2)
            # previous_distance = np.sqrt((user_position[0] - satellite_positions_at_previous_epoch[sat]["pos"][0]) ** 2
            #                             + (user_position[1] - satellite_positions_at_previous_epoch[sat]["pos"][1]) ** 2
            #                    + (user_position[2] - satellite_positions_at_previous_epoch[sat]["pos"][2]) ** 2)
            euclidian_distances[sat] = {"distance": distance, "tx_time": epoch, "pos": sat_data["pos"],
                                        "vel": sat_data["pos"] - satellite_positions_at_previous_epoch[sat]["pos"],
                                        "clock": 0}

    return euclidian_distances


def find_frequency_from_band(freq_band, prn=-1):
    if freq_band in ["E1", "L1", "S1", "S2"]:
        return 1575.42e6
    elif freq_band in ["E5", "L5"]:
        return 1176.45e6
    elif freq_band in ["L2"]:
        return 1227.6e6
    elif freq_band in ["E6"]:
        return 1279e6
    elif freq_band in ["E7"]:
        return 1207.14e6
    elif freq_band in ["R3"]:
        return 1202.025e6
    elif freq_band in ["C2", "B2"]:
        return 1561.098e6
    elif freq_band in ["C6", "B3", "C3"]:
        return 1268.52e6
    elif freq_band in ["R1"]:
        glonass_frequencies = [1, -4, 5, 6, 1, -4, 5, 6, -2, -7, 0, -1, -2, -7, 0,
                               -1, 4, -3, 3, 2, 4, -3, 3, 2, 0, 0, 0, 0, 0]
        return (1602 + 9 / 16 * glonass_frequencies[prn]) * 10**6
    elif freq_band in ["R2"]:
        glonass_frequencies = [1, -4, 5, 6, 1, -4, 5, 6, -2, -7, 0, -1, -2, -7, 0,
                               -1, 4, -3, 3, 2, 4, -3, 3, 2, 0, 0, 0, 0, 0]
        return (1246 + 7 / 16 * glonass_frequencies[prn]) * 10**6


def generate_pseudodistances(user_positions, rx_epochs, satellite_positions,
                             sampling_frequency=GENERATED_SAMPLING_FREQUENCY, label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    Depending on the noise parameters, the pseudodistances are distorted to match realistic behaviours.

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""
    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.epoch_start = satellite_positions.epoch_start
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label
    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    # generation of the satellite hardware code offsets
    satellite_hardware_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            satellite_hardware_code_errors[sat] = {}
            error = rdm.normal(0, GPS_SATELLITE_HARDWARE_INITIAL_STD)
            for freq in pseudodistances.frequencies["GPS"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_code_errors[sat][freq] = FREQUENCY_CORRELATION* error + \
                                                rdm.normal(0, np.sqrt(GPS_SATELLITE_HARDWARE_INITIAL_STD**2 *
                                                (1 - FREQUENCY_CORRELATION**2)))
                else:
                    satellite_hardware_code_errors[sat][freq] = 0
        if sat[0] == "E":
            satellite_hardware_code_errors[sat] = {}
            error = rdm.normal(0, GAL_SATELLITE_HARDWARE_INITIAL_STD)
            for freq in pseudodistances.frequencies["GAL"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_code_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                rdm.normal(0, np.sqrt(GAL_SATELLITE_HARDWARE_INITIAL_STD ** 2 *
                                                (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_code_errors[sat][freq] = 0
        if sat[0] == "R":
            satellite_hardware_code_errors[sat] = {}
            error = rdm.normal(0, GLO_SATELLITE_HARDWARE_INITIAL_STD)
            for freq in pseudodistances.frequencies["GLO"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_code_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                 rdm.normal(0, np.sqrt(GLO_SATELLITE_HARDWARE_INITIAL_STD ** 2 *
                                                 (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_code_errors[sat][freq] = 0
        if sat[0] == "C":
            satellite_hardware_code_errors[sat] = {}
            error = rdm.normal(0, BDS_SATELLITE_HARDWARE_INITIAL_STD)
            for freq in pseudodistances.frequencies["BDS"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_code_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                   rdm.normal(0, np.sqrt(BDS_SATELLITE_HARDWARE_INITIAL_STD ** 2 *
                                                   (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_code_errors[sat][freq] = 0

    # generation of the satellite hardware phase offsets
    satellite_hardware_phase_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            satellite_hardware_phase_errors[sat] = {}
            error = rdm.normal(0, GPS_SATELLITE_HARDWARE_INITIAL_STD / 100)
            for freq in pseudodistances.frequencies["GPS"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_phase_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                                rdm.normal(0, np.sqrt(
                                                                    (GPS_SATELLITE_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                                    (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_phase_errors[sat][freq] = 0
        if sat[0] == "E":
            satellite_hardware_phase_errors[sat] = {}
            error = rdm.normal(0, GAL_SATELLITE_HARDWARE_INITIAL_STD / 100)
            for freq in pseudodistances.frequencies["GAL"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_phase_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                                rdm.normal(0, np.sqrt(
                                                                    (GAL_SATELLITE_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                                    (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_phase_errors[sat][freq] = 0
        if sat[0] == "R":
            satellite_hardware_phase_errors[sat] = {}
            error = rdm.normal(0, GLO_SATELLITE_HARDWARE_INITIAL_STD / 100)
            for freq in pseudodistances.frequencies["GLO"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_phase_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                                rdm.normal(0, np.sqrt(
                                                                    (GLO_SATELLITE_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                                    (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_phase_errors[sat][freq] = 0
        if sat[0] == "C":
            satellite_hardware_phase_errors[sat] = {}
            error = rdm.normal(0, BDS_SATELLITE_HARDWARE_INITIAL_STD / 100)
            for freq in pseudodistances.frequencies["BDS"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_phase_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                                rdm.normal(0, np.sqrt(
                                                                    (BDS_SATELLITE_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                                    (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_phase_errors[sat][freq] = 0

    # generation of the receiver hardware code offsets
    receiver_hardware_code_errors = {}
    if GPS:
        error = rdm.normal(0, GPS_RECEIVER_HARDWARE_INITIAL_STD)
        for freq in pseudodistances.frequencies["GPS"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_code_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                                rdm.normal(0, np.sqrt(
                                                                    GPS_RECEIVER_HARDWARE_INITIAL_STD ** 2 *
                                                                    (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_code_errors[freq] = 0
    if GAL:
        error = rdm.normal(0, GAL_RECEIVER_HARDWARE_INITIAL_STD)
        for freq in pseudodistances.frequencies["GAL"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_code_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                      rdm.normal(0, np.sqrt(
                                                          GAL_RECEIVER_HARDWARE_INITIAL_STD ** 2 *
                                                          (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_code_errors[freq] = 0
    if GLO:
        error = rdm.normal(0, GLO_RECEIVER_HARDWARE_INITIAL_STD)
        for freq in pseudodistances.frequencies["GLO"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_code_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                      rdm.normal(0, np.sqrt(
                                                          GLO_RECEIVER_HARDWARE_INITIAL_STD ** 2 *
                                                          (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_code_errors[freq] = 0
    if BDS:
        error = rdm.normal(0, BDS_RECEIVER_HARDWARE_INITIAL_STD)
        for freq in pseudodistances.frequencies["BDS"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_code_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                      rdm.normal(0, np.sqrt(
                                                          BDS_RECEIVER_HARDWARE_INITIAL_STD ** 2 *
                                                          (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_code_errors[freq] = 0

    # generation of the receiver hardware phase offsets
    receiver_hardware_phase_errors = {}
    if GPS:
        error = rdm.normal(0, GPS_RECEIVER_HARDWARE_INITIAL_STD / 100)
        for freq in pseudodistances.frequencies["GPS"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_phase_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                      rdm.normal(0, np.sqrt(
                                                          (GPS_RECEIVER_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                          (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_phase_errors[freq] = 0
    if GAL:
        receiver_hardware_phase_errors["GAL"] = {}
        error = rdm.normal(0, GAL_RECEIVER_HARDWARE_INITIAL_STD / 100)
        for freq in pseudodistances.frequencies["GAL"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_phase_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                       rdm.normal(0, np.sqrt(
                                                           (GAL_RECEIVER_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                           (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_phase_errors[freq] = 0
    if GLO:
        receiver_hardware_phase_errors["GLO"] = {}
        error = rdm.normal(0, GLO_RECEIVER_HARDWARE_INITIAL_STD / 100)
        for freq in pseudodistances.frequencies["GLO"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_phase_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                       rdm.normal(0, np.sqrt(
                                                           (GLO_RECEIVER_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                           (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_phase_errors[freq] = 0
    if BDS:
        receiver_hardware_phase_errors["BDS"] = {}
        error = rdm.normal(0, BDS_RECEIVER_HARDWARE_INITIAL_STD / 100)
        for freq in pseudodistances.frequencies["BDS"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_phase_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                       rdm.normal(0, np.sqrt(
                                                           (BDS_RECEIVER_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                           (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_phase_errors[freq] = 0

    # generation of the receiver clock offsets
    receiver_clock_errors = {}
    if GPS:
        error = rdm.normal(0, RECEIVER_CLOCK_INITIAL_STD)
        gps_clock_error = FREQUENCY_CORRELATION * error + \
                                                       rdm.normal(0, np.sqrt(
                                                           RECEIVER_CLOCK_INITIAL_STD ** 2 *
                                                           (1 - FREQUENCY_CORRELATION ** 2)))
        for freq in pseudodistances.frequencies["GPS"]:
            if RECEIVER_CLOCK_BIAS:
                receiver_clock_errors[freq] = gps_clock_error
            else:
                receiver_clock_errors[freq] = 0
    if GAL:
        error = rdm.normal(0, RECEIVER_CLOCK_INITIAL_STD)
        gal_clock_error = FREQUENCY_CORRELATION * error + \
                                              rdm.normal(0, np.sqrt(
                                                  RECEIVER_CLOCK_INITIAL_STD ** 2 *
                                                  (1 - FREQUENCY_CORRELATION ** 2)))
        for freq in pseudodistances.frequencies["GAL"]:
            if RECEIVER_CLOCK_BIAS:
                receiver_clock_errors[freq] = gal_clock_error
            else:
                receiver_clock_errors[freq] = 0
    if GLO:
        error = rdm.normal(0, RECEIVER_CLOCK_INITIAL_STD)
        glo_clock_error = FREQUENCY_CORRELATION * error + \
                                              rdm.normal(0, np.sqrt(
                                                  RECEIVER_CLOCK_INITIAL_STD ** 2 *
                                                  (1 - FREQUENCY_CORRELATION ** 2)))
        for freq in pseudodistances.frequencies["GLO"]:
            if RECEIVER_CLOCK_BIAS:
                receiver_clock_errors[freq] = glo_clock_error
            else:
                receiver_clock_errors[freq] = 0
    if BDS:
        error = rdm.normal(0, RECEIVER_CLOCK_INITIAL_STD)
        bds_clock_error = FREQUENCY_CORRELATION * error + \
                                              rdm.normal(0, np.sqrt(
                                                  RECEIVER_CLOCK_INITIAL_STD ** 2 *
                                                  (1 - FREQUENCY_CORRELATION ** 2)))
        for freq in pseudodistances.frequencies["BDS"]:
            if RECEIVER_CLOCK_BIAS:
                receiver_clock_errors[freq] = bds_clock_error
            else:
                receiver_clock_errors[freq] = 0

    # generation of the ephemeris error
    ephemeris_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            ephemeris_errors[sat] = {}
            if EPHEMERIS_ERROR:
                ephemeris_errors[sat] = rdm.normal(0, EPHEMERIS_RANDOM_WALK_PARAMETER)
            else:
                ephemeris_errors[sat] = 0
        if sat[0] == "E":
            ephemeris_errors[sat] = {}
            if EPHEMERIS_ERROR:
                ephemeris_errors[sat] = rdm.normal(0, EPHEMERIS_RANDOM_WALK_PARAMETER)
            else:
                ephemeris_errors[sat] = 0
        if sat[0] == "R":
            ephemeris_errors[sat] = {}
            if EPHEMERIS_ERROR:
                ephemeris_errors[sat] = rdm.normal(0, EPHEMERIS_RANDOM_WALK_PARAMETER)
            else:
                ephemeris_errors[sat] = 0
        if sat[0] == "C":
            ephemeris_errors[sat] = {}
            if EPHEMERIS_ERROR:
                ephemeris_errors[sat] = rdm.normal(0, EPHEMERIS_RANDOM_WALK_PARAMETER)
            else:
                ephemeris_errors[sat] = 0

    # generation of the carrier integer error
    carrier_integer_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            carrier_integer_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if CARRIER_INTEGER_AMBIGUITY:
                    carrier_integer_errors[sat][freq] = rdm.randint(-10, 10)
                else:
                    carrier_integer_errors[sat][freq] = 0
        if sat[0] == "E":
            carrier_integer_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if CARRIER_INTEGER_AMBIGUITY:
                    carrier_integer_errors[sat][freq] = rdm.randint(-10, 10)
                else:
                    carrier_integer_errors[sat][freq] = 0
        if sat[0] == "R":
            carrier_integer_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if CARRIER_INTEGER_AMBIGUITY:
                    carrier_integer_errors[sat][freq] = rdm.randint(-10, 10)
                else:
                    carrier_integer_errors[sat][freq] = 0
        if sat[0] == "C":
            carrier_integer_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if CARRIER_INTEGER_AMBIGUITY:
                    carrier_integer_errors[sat][freq] = rdm.randint(-10, 10)
                else:
                    carrier_integer_errors[sat][freq] = 0

    # generation of the multipath code error
    multipath_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "E":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "R":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "C":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0

    # generation of the multipath phase error
    multipath_phase_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            multipath_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    multipath_phase_errors[sat][freq] = rdm.normal(0, 0.01)
                else:
                    multipath_phase_errors[sat][freq] = 0
        if sat[0] == "E":
            multipath_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    multipath_phase_errors[sat][freq] = rdm.normal(0, 0.01)
                else:
                    multipath_phase_errors[sat][freq] = 0
        if sat[0] == "R":
            multipath_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    multipath_phase_errors[sat][freq] = rdm.normal(0, 0.01)
                else:
                    multipath_phase_errors[sat][freq] = 0
        if sat[0] == "C":
            multipath_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    multipath_phase_errors[sat][freq] = rdm.normal(0, 0.01)
                else:
                    multipath_phase_errors[sat][freq] = 0

    # generation of the thermal noise code error
    thermal_noise_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            thermal_noise_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if THERMAL_NOISE:
                    thermal_noise_code_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_CODE_STD)
                else:
                    thermal_noise_code_errors[sat][freq] = 0
        if sat[0] == "E":
            thermal_noise_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if THERMAL_NOISE:
                    thermal_noise_code_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_CODE_STD)
                else:
                    thermal_noise_code_errors[sat][freq] = 0
        if sat[0] == "R":
            thermal_noise_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if THERMAL_NOISE:
                    thermal_noise_code_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_CODE_STD)
                else:
                    thermal_noise_code_errors[sat][freq] = 0
        if sat[0] == "C":
            thermal_noise_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if THERMAL_NOISE:
                    thermal_noise_code_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_CODE_STD)
                else:
                    thermal_noise_code_errors[sat][freq] = 0

    # generation of the thermal noise phase error
    thermal_noise_phase_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            thermal_noise_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if THERMAL_NOISE:
                    thermal_noise_phase_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_PHASE_STD)
                else:
                    thermal_noise_phase_errors[sat][freq] = 0
        if sat[0] == "E":
            thermal_noise_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if THERMAL_NOISE:
                    thermal_noise_phase_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_PHASE_STD)
                else:
                    thermal_noise_phase_errors[sat][freq] = 0
        if sat[0] == "R":
            thermal_noise_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if THERMAL_NOISE:
                    thermal_noise_phase_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_PHASE_STD)
                else:
                    thermal_noise_phase_errors[sat][freq] = 0
        if sat[0] == "C":
            thermal_noise_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if THERMAL_NOISE:
                    thermal_noise_phase_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_PHASE_STD)
                else:
                    thermal_noise_phase_errors[sat][freq] = 0

    # generation of the delta tropospheric zenith wet error
    delta_tropo_zenith_wet = abs(rdm.normal(0, TROPOSPHERIC_INITIAL_STD))

    # initializaion of the wind_up error container:
    wind_up = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            wind_up[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                wind_up[sat][freq] = 0
        if sat[0] == "E":
            wind_up[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                wind_up[sat][freq] = 0
        if sat[0] == "R":
            wind_up[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                wind_up[sat][freq] = 0
        if sat[0] == "C":
            wind_up[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                wind_up[sat][freq] = 0

    # generation of the error container
    sagnac = 0
    shapiro = 0
    iono_error = 0
    tropo_error = 0

    if DISPLAY_RESULTS:
        # saves of the budget error to attest the correct behaviour of the simulation
        visible_satellites = [0 for _ in range(72 * (GPS + GAL + GLO + BDS))]
        code_pseudos_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        phase_pseudos_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        doppler_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in
                              range(72 * (GPS + GAL + GLO + BDS))]
        cn0_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in
                        range(72 * (GPS + GAL + GLO + BDS))]
        iono_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        tropo_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        sagnac_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        shapiro_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        wind_up_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        receiver_clock_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        satellite_clock_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in
                                     range(72 * (GPS + GAL + GLO + BDS))]
        code_multipath_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        phase_multipath_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        code_thermal_noise_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        phase_thermal_noise_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        ephemeris_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        elevations = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        azimuths = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]

    # generation of the durations of availablilty of the satellites
    availability = {}
    for sat, _ in satellite_positions.epochs[0].items():
        availability[sat] = 0

    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance(user_positions[i], satellite_positions, epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        if i:
            delta_epoch = rx_epochs[i] - rx_epochs[i - 1]
            # generation of the new receiver clock error:
            if RECEIVER_CLOCK_BIAS:
                if GPS:
                    receiver_clock_std = np.sqrt(RECEIVER_CLOCK_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                        delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME)))
                    gps_clock_error = np.exp(-delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME) * \
                                                   receiver_clock_errors[pseudodistances.frequencies["GPS"][0]] \
                                      + rdm.normal(0, receiver_clock_std)
                    for freq in pseudodistances.frequencies["GPS"]:
                        receiver_clock_errors[freq] = gps_clock_error
                if GAL:
                    receiver_clock_std = np.sqrt(RECEIVER_CLOCK_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                        delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME)))
                    gal_clock_error = np.exp(-delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME) * \
                                                   receiver_clock_errors[pseudodistances.frequencies["GAL"][0]] \
                                      + rdm.normal(0, receiver_clock_std)
                    for freq in pseudodistances.frequencies["GAL"]:
                        receiver_clock_errors[freq] = gal_clock_error
                if GLO:
                    receiver_clock_std = np.sqrt(RECEIVER_CLOCK_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                        delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME)))
                    glo_clock_error = np.exp(-delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME) * \
                                                   receiver_clock_errors[pseudodistances.frequencies["GLO"][0]] \
                                      + rdm.normal(0, receiver_clock_std)
                    for freq in pseudodistances.frequencies["GLO"]:
                        receiver_clock_errors[freq] = glo_clock_error
                if BDS:
                    receiver_clock_std = np.sqrt(RECEIVER_CLOCK_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                        delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME)))
                    bds_clock_error = np.exp(-delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME) * \
                                                   receiver_clock_errors[pseudodistances.frequencies["BDS"][0]] \
                                      + rdm.normal(0, receiver_clock_std)
                    for freq in pseudodistances.frequencies["BDS"]:
                        receiver_clock_errors[freq] = bds_clock_error

        for sat, sat_pos in euclidian_distances.items():
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if EPHEMERIS_ERROR and availability[sat]:
                # generation of the new ephemeris error:
                ephemeris_std = np.sqrt(EPHEMERIS_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                delta_epoch / EPHEMERIS_CORRELATION_TIME)))
                ephemeris_errors[sat] = np.exp(-delta_epoch / EPHEMERIS_CORRELATION_TIME) * \
                                        ephemeris_errors[sat] + rdm.normal(0, ephemeris_std)

            if TROPOSPHERIC_ERROR and availability[sat]:
                # generation of the new delta tropospheric vertical wet error:
                tropospheric_wet_std = np.sqrt(TROPOSPHERIC_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                delta_epoch / TROPOSPHERIC_CORRELATION_TIME)))
                delta_tropo_zenith_wet = np.exp(-delta_epoch / TROPOSPHERIC_CORRELATION_TIME) * \
                                         delta_tropo_zenith_wet + rdm.normal(0, tropospheric_wet_std)

            for freq in frequencies:

                if availability[sat]:

                    # generation of the new multipath errors:
                    if MULTIPATH_ERROR:
                        # generation of the new multipath code error:
                        multipath_code_std = np.sqrt(MULTIPATH_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                        distance(user_positions[i], user_positions[i-1]) / MULTIPATH_CORRELATION_DISTANCE)))
                        multipath_code_errors[sat][freq] = np.exp(-distance(user_positions[i], user_positions[i-1]) /
                                        MULTIPATH_CORRELATION_DISTANCE) * multipath_code_errors[sat][freq] \
                                        + rdm.normal(0, multipath_code_std)
                        # generation of the new multipath phase error:
                        multipath_phase_std = np.sqrt((MULTIPATH_RANDOM_WALK_PARAMETER / 100) ** 2 * (1 - np.exp(-2 *
                                        distance(user_positions[i], user_positions[i-1]) / MULTIPATH_CORRELATION_DISTANCE)))
                        multipath_phase_errors[sat][freq] = np.exp(-distance(user_positions[i], user_positions[i - 1]) /
                                        MULTIPATH_CORRELATION_DISTANCE) * multipath_phase_errors[sat][freq] \
                                        + rdm.normal(0, multipath_phase_std)
                    # generation of the new thermal noise errors:
                    if THERMAL_NOISE:
                        # generation of the new thermal noise code errors:
                        thermal_noise_code_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_CODE_STD)
                        # generation of the new thermal noise phase error:
                        thermal_noise_phase_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_PHASE_STD)

                if SAGNAC_EFFECT:
                    # determination of the Sagnac error
                    sagnac = sagnac_error(sat_pos["pos"], sat_pos["vel"])
                if CARRIER_WIND_UP:
                    # determination of the carrier phase wind-up error :
                    frequency = find_frequency_from_band(freq)
                    wind_up[sat][freq] = wind_up_error(user_positions[i], sat_pos["pos"], rx_epochs[i],
                                                       frequency, wind_up[sat][freq])
                if SHAPIRO_EFFECT:
                    # determination of the Shapiro error
                    shapiro = shapiro_error(user_positions[i], sat_pos["pos"])
                if IONOSPHERIC_ERROR:
                    # determination of the ionospheric error
                    frequency = find_frequency_from_band(freq)
                    elevation = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])[0]
                    iono_error = ionospheric_error(frequency, elevation)
                if TROPOSPHERIC_ERROR:
                    # determination of the tropospheric delay
                    elevation = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])[0]
                    day_of_year = (rx_epochs[i] / 86400 + 6) % 365.25
                    tropo_error = tropospheric_error(user_positions[i], elevation,
                                                     delta_tropo_zenith_wet, day_of_year)

                # Doppler figure computation :
                if i:
                    user_velocity = (user_positions[i] - user_positions[i - 1]) * sampling_frequency
                else:
                    user_velocity = (user_positions[i + 1] - user_positions[i]) * sampling_frequency

                doppler = frequency * np.dot(user_velocity - sat_pos["vel"],
                                             (sat_pos["pos"] - user_positions[i]) / sat_pos["distance"]) / C

                # C/N0 estimation
                cn0 = 113 - sat_pos["distance"] * 3.018e-6

                # generation of the new pseudodistances for the given satellite, frequency and epoch
                pseudodistances.epochs[i][sat][freq] = {"code": sat_pos["distance"] + ephemeris_errors[sat]
                        + receiver_hardware_code_errors[freq] + satellite_hardware_code_errors[sat][freq]
                        + receiver_clock_errors[freq] - sat_pos["clock"]
                        + multipath_code_errors[sat][freq] + thermal_noise_code_errors[sat][freq]
                        - sagnac - shapiro + iono_error + tropo_error,
                        "phase": sat_pos["distance"] + ephemeris_errors[sat]
                        + receiver_hardware_phase_errors[freq] + satellite_hardware_phase_errors[sat][freq]
                        + receiver_clock_errors[freq] - sat_pos["clock"]
                        + multipath_phase_errors[sat][freq] + thermal_noise_phase_errors[sat][freq]
                        + carrier_integer_errors[sat][freq] + wind_up[sat][freq] - sagnac - shapiro
                        - iono_error + tropo_error,
                        "doppler": doppler, "CN0": cn0}
                pseudodistances.nb_pseudoranges += 1

                if DISPLAY_RESULTS:
                    pos = save_pos(sat, freq)
                    visible_satellites[pos] = 1
                    code_pseudos_save[pos][i] = pseudodistances.epochs[i][sat][freq]["code"]
                    phase_pseudos_save[pos][i] = pseudodistances.epochs[i][sat][freq]["phase"]
                    doppler_save[pos][i] = pseudodistances.epochs[i][sat][freq]["doppler"]
                    cn0_save[pos][i] = pseudodistances.epochs[i][sat][freq]["CN0"]
                    iono_error_save[pos][i] = iono_error
                    tropo_error_save[pos][i] = tropo_error
                    ephemeris_error_save[pos][i] = ephemeris_errors[sat]
                    receiver_clock_error_save[pos][i] = receiver_clock_errors[freq]
                    satellite_clock_error_save[pos][i] = sat_pos["clock"]
                    code_multipath_error_save[pos][i] = multipath_code_errors[sat][freq]
                    phase_multipath_error_save[pos][i] = multipath_phase_errors[sat][freq]
                    code_thermal_noise_error_save[pos][i] = thermal_noise_code_errors[sat][freq]
                    phase_thermal_noise_error_save[pos][i] = thermal_noise_phase_errors[sat][freq]
                    sagnac_error_save[pos][i] = sagnac
                    shapiro_error_save[pos][i] = shapiro
                    wind_up_error_save[pos][i] = wind_up[sat][freq]
                    elevation, azimuth = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])
                    elevations[pos][i] = elevation
                    azimuths[pos][i] = azimuth

        visible_satellites_at_epoch = []
        for sat, _ in euclidian_distances.items():
            visible_satellites_at_epoch.append(sat)
        for sat, _ in satellite_positions.epochs[0].items():
            if sat in visible_satellites_at_epoch:
                availability[sat] += 1
            else:
                availability[sat] = 0

    if DISPLAY_RESULTS:
        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], code_pseudos_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Code pseudodistances of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("pseudoranges (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], phase_pseudos_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Phase pseudodistances of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("pseudoranges (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], iono_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Ionospheric errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], tropo_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Tropospheric errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], ephemeris_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Ephemeris errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        prn_to_display = 4
        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j] and j == prn_to_display:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], receiver_clock_error_save[j], "-k", linewidth=2)
        plt.title("Receiver clock error")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], satellite_clock_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Satellite clock errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], code_multipath_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Code multipath errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], phase_multipath_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Phase multipath errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], code_thermal_noise_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Code thermal noise errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], phase_thermal_noise_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Phase thermal noise errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], sagnac_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Sagnac effect errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], shapiro_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Shapiro effect errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], wind_up_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Wind-up errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        #plt.savefig("wind_up_testsave", dpi=500)
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], doppler_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Doppler shifts of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("Doppler shift (Hz)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], cn0_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Carrier to Noise ratio of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("C/N0 (dB.Hz)")
        plt.show()

        # # plots the apparent position of the satellites in the sky
        reference_azimuths = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        reference_elevations = [0, 15, 30, 45, 60, 75]
        nb_points_reference = 360
        nb_comet = int(1200 / (rx_epochs[1] - rx_epochs[0])) + 2
        plt.figure()

        for epoch in range(2, pseudodistances.nb_epochs):
            plt.clf()
            for el in reference_elevations:
                radius = ((90 - el) / 90) ** 1.05
                x = np.linspace(-radius, radius, nb_points_reference)
                y = np.sqrt(radius ** 2 - np.power(x, 2))
                plt.plot(x, y, "--k", alpha=0.5, linewidth=0.5)
                plt.plot(x, -y, "--k", alpha=0.5, linewidth=0.5)
                if el:
                    plt.text(0.02, radius + 0.02, str(el) + "°")
            for az in reference_azimuths:
                theta = (90 - az) * DEG2RAD
                plt.plot([0, np.cos(theta)], [0, np.sin(theta)], "--k", alpha=0.5, linewidth=0.5)
                plt.text(np.cos(theta) * 1.05, np.sin(theta) * 1.05, str(az) + "°", rotation=theta * RAD2DEG - 90,
                         ha="center", va="center")
            for j in range(72 * (GPS + GAL + GLO + BDS)):
                x = []
                y = []
                sat = save_sat(j)
                if sat[0] == "G":
                    col = "-b"
                elif sat[0] == "E":
                    col = "-r"
                elif sat[0] == "R":
                    col = "-g"
                else:
                    col = "orange"
                for k in range(min(epoch + 1, nb_comet)):
                    if azimuths[j][epoch - k] is not None:
                        x.append(np.cos((np.pi/2 - azimuths[j][epoch - k])) *
                                 ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                        y.append(np.sin((np.pi/2 - azimuths[j][epoch - k])) *
                                 ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                        if len(x) > 1:
                            plt.plot([x[-2], x[-1]], [y[-2], y[-1]], col, linewidth=(1 - k/nb_comet)*3)
                if len(x):
                    plt.text(x[0], y[0], sat[0:4], fontweight="semibold")
            plt.xlim([-1.3, 1.3])
            plt.axis("equal")
            plt.xticks([], [])
            plt.yticks([], [])
            #mng = plt.get_current_fig_manager() # to uncomment to plot full screen
            #mng.window.state('zoomed') # to uncomment to plot full screen
            plt.pause(0.0006)
        plt.show()

    return pseudodistances


def generate_pseudodistances_with_meaconer(user_positions, rx_epochs, satellite_positions, meaconer,
                                           sampling_frequency=GENERATED_SAMPLING_FREQUENCY, label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    Depending on the noise parameters, the pseudodistances are distorted to match realistic behaviours.

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""
    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.epoch_start = satellite_positions.epoch_start
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label
    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    # generation of the satellite hardware code offsets
    satellite_hardware_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            satellite_hardware_code_errors[sat] = {}
            error = rdm.normal(0, GPS_SATELLITE_HARDWARE_INITIAL_STD)
            for freq in pseudodistances.frequencies["GPS"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_code_errors[sat][freq] = FREQUENCY_CORRELATION* error + \
                                                rdm.normal(0, np.sqrt(GPS_SATELLITE_HARDWARE_INITIAL_STD**2 *
                                                (1 - FREQUENCY_CORRELATION**2)))
                else:
                    satellite_hardware_code_errors[sat][freq] = 0
        if sat[0] == "E":
            satellite_hardware_code_errors[sat] = {}
            error = rdm.normal(0, GAL_SATELLITE_HARDWARE_INITIAL_STD)
            for freq in pseudodistances.frequencies["GAL"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_code_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                rdm.normal(0, np.sqrt(GAL_SATELLITE_HARDWARE_INITIAL_STD ** 2 *
                                                (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_code_errors[sat][freq] = 0
        if sat[0] == "R":
            satellite_hardware_code_errors[sat] = {}
            error = rdm.normal(0, GLO_SATELLITE_HARDWARE_INITIAL_STD)
            for freq in pseudodistances.frequencies["GLO"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_code_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                 rdm.normal(0, np.sqrt(GLO_SATELLITE_HARDWARE_INITIAL_STD ** 2 *
                                                 (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_code_errors[sat][freq] = 0
        if sat[0] == "C":
            satellite_hardware_code_errors[sat] = {}
            error = rdm.normal(0, BDS_SATELLITE_HARDWARE_INITIAL_STD)
            for freq in pseudodistances.frequencies["BDS"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_code_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                   rdm.normal(0, np.sqrt(BDS_SATELLITE_HARDWARE_INITIAL_STD ** 2 *
                                                   (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_code_errors[sat][freq] = 0

    # generation of the satellite hardware phase offsets
    satellite_hardware_phase_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            satellite_hardware_phase_errors[sat] = {}
            error = rdm.normal(0, GPS_SATELLITE_HARDWARE_INITIAL_STD / 100)
            for freq in pseudodistances.frequencies["GPS"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_phase_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                                rdm.normal(0, np.sqrt(
                                                                    (GPS_SATELLITE_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                                    (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_phase_errors[sat][freq] = 0
        if sat[0] == "E":
            satellite_hardware_phase_errors[sat] = {}
            error = rdm.normal(0, GAL_SATELLITE_HARDWARE_INITIAL_STD / 100)
            for freq in pseudodistances.frequencies["GAL"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_phase_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                                rdm.normal(0, np.sqrt(
                                                                    (GAL_SATELLITE_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                                    (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_phase_errors[sat][freq] = 0
        if sat[0] == "R":
            satellite_hardware_phase_errors[sat] = {}
            error = rdm.normal(0, GLO_SATELLITE_HARDWARE_INITIAL_STD / 100)
            for freq in pseudodistances.frequencies["GLO"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_phase_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                                rdm.normal(0, np.sqrt(
                                                                    (GLO_SATELLITE_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                                    (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_phase_errors[sat][freq] = 0
        if sat[0] == "C":
            satellite_hardware_phase_errors[sat] = {}
            error = rdm.normal(0, BDS_SATELLITE_HARDWARE_INITIAL_STD / 100)
            for freq in pseudodistances.frequencies["BDS"]:
                if SATELLITE_HARDWARE_BIAS:
                    satellite_hardware_phase_errors[sat][freq] = FREQUENCY_CORRELATION * error + \
                                                                rdm.normal(0, np.sqrt(
                                                                    (BDS_SATELLITE_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                                    (1 - FREQUENCY_CORRELATION ** 2)))
                else:
                    satellite_hardware_phase_errors[sat][freq] = 0

    # generation of the receiver hardware code offsets
    receiver_hardware_code_errors = {}
    if GPS:
        error = rdm.normal(0, GPS_RECEIVER_HARDWARE_INITIAL_STD)
        for freq in pseudodistances.frequencies["GPS"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_code_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                                rdm.normal(0, np.sqrt(
                                                                    GPS_RECEIVER_HARDWARE_INITIAL_STD ** 2 *
                                                                    (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_code_errors[freq] = 0
    if GAL:
        error = rdm.normal(0, GAL_RECEIVER_HARDWARE_INITIAL_STD)
        for freq in pseudodistances.frequencies["GAL"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_code_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                      rdm.normal(0, np.sqrt(
                                                          GAL_RECEIVER_HARDWARE_INITIAL_STD ** 2 *
                                                          (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_code_errors[freq] = 0
    if GLO:
        error = rdm.normal(0, GLO_RECEIVER_HARDWARE_INITIAL_STD)
        for freq in pseudodistances.frequencies["GLO"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_code_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                      rdm.normal(0, np.sqrt(
                                                          GLO_RECEIVER_HARDWARE_INITIAL_STD ** 2 *
                                                          (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_code_errors[freq] = 0
    if BDS:
        error = rdm.normal(0, BDS_RECEIVER_HARDWARE_INITIAL_STD)
        for freq in pseudodistances.frequencies["BDS"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_code_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                      rdm.normal(0, np.sqrt(
                                                          BDS_RECEIVER_HARDWARE_INITIAL_STD ** 2 *
                                                          (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_code_errors[freq] = 0

    # generation of the receiver hardware phase offsets
    receiver_hardware_phase_errors = {}
    if GPS:
        error = rdm.normal(0, GPS_RECEIVER_HARDWARE_INITIAL_STD / 100)
        for freq in pseudodistances.frequencies["GPS"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_phase_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                      rdm.normal(0, np.sqrt(
                                                          (GPS_RECEIVER_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                          (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_phase_errors[freq] = 0
    if GAL:
        receiver_hardware_phase_errors["GAL"] = {}
        error = rdm.normal(0, GAL_RECEIVER_HARDWARE_INITIAL_STD / 100)
        for freq in pseudodistances.frequencies["GAL"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_phase_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                       rdm.normal(0, np.sqrt(
                                                           (GAL_RECEIVER_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                           (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_phase_errors[freq] = 0
    if GLO:
        receiver_hardware_phase_errors["GLO"] = {}
        error = rdm.normal(0, GLO_RECEIVER_HARDWARE_INITIAL_STD / 100)
        for freq in pseudodistances.frequencies["GLO"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_phase_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                       rdm.normal(0, np.sqrt(
                                                           (GLO_RECEIVER_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                           (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_phase_errors[freq] = 0
    if BDS:
        receiver_hardware_phase_errors["BDS"] = {}
        error = rdm.normal(0, BDS_RECEIVER_HARDWARE_INITIAL_STD / 100)
        for freq in pseudodistances.frequencies["BDS"]:
            if RECEIVER_HARDWARE_BIAS:
                receiver_hardware_phase_errors[freq] = FREQUENCY_CORRELATION * error + \
                                                       rdm.normal(0, np.sqrt(
                                                           (BDS_RECEIVER_HARDWARE_INITIAL_STD / 100) ** 2 *
                                                           (1 - FREQUENCY_CORRELATION ** 2)))
            else:
                receiver_hardware_phase_errors[freq] = 0

    # generation of the receiver clock offsets
    receiver_clock_errors = {}
    if GPS:
        error = rdm.normal(0, RECEIVER_CLOCK_INITIAL_STD)
        gps_clock_error = FREQUENCY_CORRELATION * error + \
                                                       rdm.normal(0, np.sqrt(
                                                           RECEIVER_CLOCK_INITIAL_STD ** 2 *
                                                           (1 - FREQUENCY_CORRELATION ** 2)))
        for freq in pseudodistances.frequencies["GPS"]:
            if RECEIVER_CLOCK_BIAS:
                receiver_clock_errors[freq] = gps_clock_error
            else:
                receiver_clock_errors[freq] = 0
    if GAL:
        error = rdm.normal(0, RECEIVER_CLOCK_INITIAL_STD)
        gal_clock_error = FREQUENCY_CORRELATION * error + \
                                              rdm.normal(0, np.sqrt(
                                                  RECEIVER_CLOCK_INITIAL_STD ** 2 *
                                                  (1 - FREQUENCY_CORRELATION ** 2)))
        for freq in pseudodistances.frequencies["GAL"]:
            if RECEIVER_CLOCK_BIAS:
                receiver_clock_errors[freq] = gal_clock_error
            else:
                receiver_clock_errors[freq] = 0
    if GLO:
        error = rdm.normal(0, RECEIVER_CLOCK_INITIAL_STD)
        glo_clock_error = FREQUENCY_CORRELATION * error + \
                                              rdm.normal(0, np.sqrt(
                                                  RECEIVER_CLOCK_INITIAL_STD ** 2 *
                                                  (1 - FREQUENCY_CORRELATION ** 2)))
        for freq in pseudodistances.frequencies["GLO"]:
            if RECEIVER_CLOCK_BIAS:
                receiver_clock_errors[freq] = glo_clock_error
            else:
                receiver_clock_errors[freq] = 0
    if BDS:
        error = rdm.normal(0, RECEIVER_CLOCK_INITIAL_STD)
        bds_clock_error = FREQUENCY_CORRELATION * error + \
                                              rdm.normal(0, np.sqrt(
                                                  RECEIVER_CLOCK_INITIAL_STD ** 2 *
                                                  (1 - FREQUENCY_CORRELATION ** 2)))
        for freq in pseudodistances.frequencies["BDS"]:
            if RECEIVER_CLOCK_BIAS:
                receiver_clock_errors[freq] = bds_clock_error
            else:
                receiver_clock_errors[freq] = 0

    # generation of the ephemeris error
    ephemeris_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            ephemeris_errors[sat] = {}
            if EPHEMERIS_ERROR:
                ephemeris_errors[sat] = rdm.normal(0, EPHEMERIS_RANDOM_WALK_PARAMETER)
            else:
                ephemeris_errors[sat] = 0
        if sat[0] == "E":
            ephemeris_errors[sat] = {}
            if EPHEMERIS_ERROR:
                ephemeris_errors[sat] = rdm.normal(0, EPHEMERIS_RANDOM_WALK_PARAMETER)
            else:
                ephemeris_errors[sat] = 0
        if sat[0] == "R":
            ephemeris_errors[sat] = {}
            if EPHEMERIS_ERROR:
                ephemeris_errors[sat] = rdm.normal(0, EPHEMERIS_RANDOM_WALK_PARAMETER)
            else:
                ephemeris_errors[sat] = 0
        if sat[0] == "C":
            ephemeris_errors[sat] = {}
            if EPHEMERIS_ERROR:
                ephemeris_errors[sat] = rdm.normal(0, EPHEMERIS_RANDOM_WALK_PARAMETER)
            else:
                ephemeris_errors[sat] = 0

    # generation of the carrier integer error
    carrier_integer_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            carrier_integer_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if CARRIER_INTEGER_AMBIGUITY:
                    carrier_integer_errors[sat][freq] = rdm.randint(-10, 10)
                else:
                    carrier_integer_errors[sat][freq] = 0
        if sat[0] == "E":
            carrier_integer_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if CARRIER_INTEGER_AMBIGUITY:
                    carrier_integer_errors[sat][freq] = rdm.randint(-10, 10)
                else:
                    carrier_integer_errors[sat][freq] = 0
        if sat[0] == "R":
            carrier_integer_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if CARRIER_INTEGER_AMBIGUITY:
                    carrier_integer_errors[sat][freq] = rdm.randint(-10, 10)
                else:
                    carrier_integer_errors[sat][freq] = 0
        if sat[0] == "C":
            carrier_integer_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if CARRIER_INTEGER_AMBIGUITY:
                    carrier_integer_errors[sat][freq] = rdm.randint(-10, 10)
                else:
                    carrier_integer_errors[sat][freq] = 0

    # generation of the multipath code error
    multipath_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "E":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "R":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "C":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0

    # generation of the multipath phase error
    multipath_phase_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            multipath_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    multipath_phase_errors[sat][freq] = rdm.normal(0, 0.01)
                else:
                    multipath_phase_errors[sat][freq] = 0
        if sat[0] == "E":
            multipath_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    multipath_phase_errors[sat][freq] = rdm.normal(0, 0.01)
                else:
                    multipath_phase_errors[sat][freq] = 0
        if sat[0] == "R":
            multipath_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    multipath_phase_errors[sat][freq] = rdm.normal(0, 0.01)
                else:
                    multipath_phase_errors[sat][freq] = 0
        if sat[0] == "C":
            multipath_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    multipath_phase_errors[sat][freq] = rdm.normal(0, 0.01)
                else:
                    multipath_phase_errors[sat][freq] = 0

    # generation of the thermal noise code error
    thermal_noise_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            thermal_noise_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if THERMAL_NOISE:
                    thermal_noise_code_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_CODE_STD)
                else:
                    thermal_noise_code_errors[sat][freq] = 0
        if sat[0] == "E":
            thermal_noise_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if THERMAL_NOISE:
                    thermal_noise_code_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_CODE_STD)
                else:
                    thermal_noise_code_errors[sat][freq] = 0
        if sat[0] == "R":
            thermal_noise_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if THERMAL_NOISE:
                    thermal_noise_code_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_CODE_STD)
                else:
                    thermal_noise_code_errors[sat][freq] = 0
        if sat[0] == "C":
            thermal_noise_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if THERMAL_NOISE:
                    thermal_noise_code_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_CODE_STD)
                else:
                    thermal_noise_code_errors[sat][freq] = 0

    # generation of the thermal noise phase error
    thermal_noise_phase_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            thermal_noise_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if THERMAL_NOISE:
                    thermal_noise_phase_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_PHASE_STD)
                else:
                    thermal_noise_phase_errors[sat][freq] = 0
        if sat[0] == "E":
            thermal_noise_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if THERMAL_NOISE:
                    thermal_noise_phase_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_PHASE_STD)
                else:
                    thermal_noise_phase_errors[sat][freq] = 0
        if sat[0] == "R":
            thermal_noise_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if THERMAL_NOISE:
                    thermal_noise_phase_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_PHASE_STD)
                else:
                    thermal_noise_phase_errors[sat][freq] = 0
        if sat[0] == "C":
            thermal_noise_phase_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if THERMAL_NOISE:
                    thermal_noise_phase_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_PHASE_STD)
                else:
                    thermal_noise_phase_errors[sat][freq] = 0

    # generation of the delta tropospheric zenith wet error
    delta_tropo_zenith_wet = abs(rdm.normal(0, TROPOSPHERIC_INITIAL_STD))

    # initializaion of the wind_up error container:
    wind_up = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            wind_up[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                wind_up[sat][freq] = 0
        if sat[0] == "E":
            wind_up[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                wind_up[sat][freq] = 0
        if sat[0] == "R":
            wind_up[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                wind_up[sat][freq] = 0
        if sat[0] == "C":
            wind_up[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                wind_up[sat][freq] = 0

    # initiallization of the delta_tau of the tracking loops
    previous_delta_tau = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat != "time":
            previous_delta_tau[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                previous_delta_tau[sat][freq] = None

    # generation of the error container
    sagnac = 0
    shapiro = 0
    iono_error = 0
    tropo_error = 0

    if DISPLAY_RESULTS:
        # saves of the budget error to attest the correct behaviour of the simulation
        visible_satellites = [0 for _ in range(72 * (GPS + GAL + GLO + BDS))]
        code_pseudos_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        phase_pseudos_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        iono_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        tropo_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        sagnac_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        shapiro_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        wind_up_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        receiver_clock_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        satellite_clock_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in
                                     range(72 * (GPS + GAL + GLO + BDS))]
        code_multipath_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        phase_multipath_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        code_thermal_noise_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        phase_thermal_noise_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        ephemeris_error_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        elevations = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
        azimuths = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]

    # generation of the durations of availablilty of the satellites
    availability = {}
    for sat, _ in satellite_positions.epochs[0].items():
        availability[sat] = 0

    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance(user_positions[i], satellite_positions, epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        if i:
            delta_epoch = rx_epochs[i] - rx_epochs[i - 1]
            # generation of the new receiver clock error:
            if RECEIVER_CLOCK_BIAS:
                if GPS:
                    receiver_clock_std = np.sqrt(RECEIVER_CLOCK_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                        delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME)))
                    gps_clock_error = np.exp(-delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME) * \
                                                   receiver_clock_errors[pseudodistances.frequencies["GPS"][0]] \
                                      + rdm.normal(0, receiver_clock_std)
                    for freq in pseudodistances.frequencies["GPS"]:
                        receiver_clock_errors[freq] = gps_clock_error
                if GAL:
                    receiver_clock_std = np.sqrt(RECEIVER_CLOCK_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                        delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME)))
                    gal_clock_error = np.exp(-delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME) * \
                                                   receiver_clock_errors[pseudodistances.frequencies["GAL"][0]] \
                                      + rdm.normal(0, receiver_clock_std)
                    for freq in pseudodistances.frequencies["GAL"]:
                        receiver_clock_errors[freq] = gal_clock_error
                if GLO:
                    receiver_clock_std = np.sqrt(RECEIVER_CLOCK_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                        delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME)))
                    glo_clock_error = np.exp(-delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME) * \
                                                   receiver_clock_errors[pseudodistances.frequencies["GLO"][0]] \
                                      + rdm.normal(0, receiver_clock_std)
                    for freq in pseudodistances.frequencies["GLO"]:
                        receiver_clock_errors[freq] = glo_clock_error
                if BDS:
                    receiver_clock_std = np.sqrt(RECEIVER_CLOCK_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                        delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME)))
                    bds_clock_error = np.exp(-delta_epoch / RECEIVER_CLOCK_CORRELATION_TIME) * \
                                                   receiver_clock_errors[pseudodistances.frequencies["BDS"][0]] \
                                      + rdm.normal(0, receiver_clock_std)
                    for freq in pseudodistances.frequencies["BDS"]:
                        receiver_clock_errors[freq] = bds_clock_error

        for sat, sat_pos in euclidian_distances.items():
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if EPHEMERIS_ERROR and availability[sat]:
                # generation of the new ephemeris error:
                ephemeris_std = np.sqrt(EPHEMERIS_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                delta_epoch / EPHEMERIS_CORRELATION_TIME)))
                ephemeris_errors[sat] = np.exp(-delta_epoch / EPHEMERIS_CORRELATION_TIME) * \
                                        ephemeris_errors[sat] + rdm.normal(0, ephemeris_std)

            if TROPOSPHERIC_ERROR and availability[sat]:
                # generation of the new delta tropospheric vertical wet error:
                tropospheric_wet_std = np.sqrt(TROPOSPHERIC_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                                                delta_epoch / TROPOSPHERIC_CORRELATION_TIME)))
                delta_tropo_zenith_wet = np.exp(-delta_epoch / TROPOSPHERIC_CORRELATION_TIME) * \
                                         delta_tropo_zenith_wet + rdm.normal(0, tropospheric_wet_std)

            for freq in frequencies:

                if availability[sat]:

                    # generation of the new multipath errors:
                    if MULTIPATH_ERROR:
                        # generation of the new multipath code error:
                        multipath_code_std = np.sqrt(MULTIPATH_RANDOM_WALK_PARAMETER ** 2 * (1 - np.exp(-2 *
                                        distance(user_positions[i], user_positions[i-1]) / MULTIPATH_CORRELATION_DISTANCE)))
                        multipath_code_errors[sat][freq] = np.exp(-distance(user_positions[i], user_positions[i-1]) /
                                        MULTIPATH_CORRELATION_DISTANCE) * multipath_code_errors[sat][freq] \
                                        + rdm.normal(0, multipath_code_std)
                        # generation of the new multipath phase error:
                        multipath_phase_std = np.sqrt((MULTIPATH_RANDOM_WALK_PARAMETER / 100) ** 2 * (1 - np.exp(-2 *
                                        distance(user_positions[i], user_positions[i-1]) / MULTIPATH_CORRELATION_DISTANCE)))
                        multipath_phase_errors[sat][freq] = np.exp(-distance(user_positions[i], user_positions[i - 1]) /
                                        MULTIPATH_CORRELATION_DISTANCE) * multipath_phase_errors[sat][freq] \
                                        + rdm.normal(0, multipath_phase_std)
                    # generation of the new thermal noise errors:
                    if THERMAL_NOISE:
                        # generation of the new thermal noise code errors:
                        thermal_noise_code_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_CODE_STD)
                        # generation of the new thermal noise phase error:
                        thermal_noise_phase_errors[sat][freq] = rdm.normal(0, THERMAL_NOISE_PHASE_STD)

                if SAGNAC_EFFECT:
                    # determination of the Sagnac error
                    sagnac = sagnac_error(sat_pos["pos"], sat_pos["vel"])
                if CARRIER_WIND_UP:
                    # determination of the carrier phase wind-up error :
                    frequency = find_frequency_from_band(freq)
                    wind_up[sat][freq] = wind_up_error(user_positions[i], sat_pos["pos"], rx_epochs[i],
                                                       frequency, wind_up[sat][freq])
                if SHAPIRO_EFFECT:
                    # determination of the Shapiro error
                    shapiro = shapiro_error(user_positions[i], sat_pos["pos"])
                if IONOSPHERIC_ERROR:
                    # determination of the ionospheric error
                    frequency = find_frequency_from_band(freq)
                    elevation = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])[0]
                    iono_error = ionospheric_error(frequency, elevation)
                if TROPOSPHERIC_ERROR:
                    # determination of the tropospheric delay
                    elevation = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])[0]
                    day_of_year = (rx_epochs[i] / 86400 + 6) % 365.25
                    tropo_error = tropospheric_error(user_positions[i], elevation,
                                                     delta_tropo_zenith_wet, day_of_year)

                # Doppler figure computation :
                if i:
                    user_velocity = (user_positions[i] - user_positions[i - 1]) * sampling_frequency
                else:
                    user_velocity = (user_positions[i + 1] - user_positions[i]) * sampling_frequency

                doppler = frequency * np.dot(user_velocity - sat_pos["vel"],
                                             (sat_pos["pos"] - user_positions[i]) / sat_pos["distance"]) / C

                # C/N0 estimation
                cn0 = 113 - sat_pos["distance"] * 3.018e-6

                # addition of the meaconer's induced bias
                delta_tau = np.linalg.norm(sat_pos["pos"] - np.squeeze(np.transpose(meaconer.ecef_position))) / C - \
                            sat_pos["distance"] / C + meaconer.delay * 1e9 + \
                            np.linalg.norm(user_positions[i] - np.squeeze(np.transpose(meaconer.ecef_position))) / C
                frequency = find_frequency_from_band(freq)
                lamda = C / frequency
                delta_theta = np.mod(delta_tau, lamda) * 2 * np.pi / lamda

                elevation_meaconer = conversion.ECEF2elevation_azimuth(user_positions[i], np.squeeze(
                    np.transpose(meaconer.ecef_position)))[0]
                signal_attenuation = (lamda / 4 / np.pi / np.linalg.norm(
                    user_positions[i] - np.squeeze(np.transpose(meaconer.ecef_position)))) ** 2 \
                                     * maximum_antenna_gain(elevation_meaconer)
                integration_time = 0.02  # in seconds
                inter_chip_spacing = 1 / 1023000  # in seconds
                bandwidth = 5e6  # in Hertz, bandwidth of the front-end filter
                chip_period = 1 / 1023000  # inverse of the chipping rate
                autocorr_function = lambda tau: meaconer_bias_estimation.R_2(tau, bandwidth, chip_period)
                meaconer_bias = meaconer_bias_estimation.compute_meaconer_bias(
                    meaconer_bias_estimation.discriminator_NEMLE, meaconer_bias_estimation.discriminator_atan2,
                    integration_time, doppler + frequency, autocorr_function, inter_chip_spacing,
                    1,
                    [delta_tau],
                    [meaconer.delay * 1e-9],
                    [signal_attenuation],
                    [10 ** (meaconer.gain / 10)], 1, chip_period,
                    previous_delta_tau[sat][freq])
                old_meaconer_bias = old_meaconer_bias = meaconer_bias_estimation_old_version.compute_meaconer_bias(
                        delta_tau, meaconer.delay * 1e-9,
                                                                                   signal_attenuation,
                                                                                   10 ** (meaconer.gain / 10),
                                                                                   previous_delta_tau[sat][freq])
                previous_delta_tau[sat][freq] = delta_tau

                # generation of the new pseudodistances for the given satellite, frequency and epoch
                pseudodistances.epochs[i][sat][freq] = {"code": sat_pos["distance"] + ephemeris_errors[sat]
                        + receiver_hardware_code_errors[freq] + satellite_hardware_code_errors[sat][freq]
                        + receiver_clock_errors[freq] - sat_pos["clock"]
                        + multipath_code_errors[sat][freq] + thermal_noise_code_errors[sat][freq]
                        - sagnac - shapiro + iono_error + tropo_error + meaconer_bias[0],
                        "phase": sat_pos["distance"] + ephemeris_errors[sat]
                        + receiver_hardware_phase_errors[freq] + satellite_hardware_phase_errors[sat][freq]
                        + receiver_clock_errors[freq] - sat_pos["clock"]
                        + multipath_phase_errors[sat][freq] + thermal_noise_phase_errors[sat][freq]
                        + carrier_integer_errors[sat][freq] + wind_up[sat][freq] - sagnac - shapiro
                        - iono_error + tropo_error + meaconer_bias[1] * lamda,
                        "doppler": doppler, "CN0": cn0}
                pseudodistances.nb_pseudoranges += 1

                if DISPLAY_RESULTS:
                    pos = save_pos(sat, freq)
                    visible_satellites[pos] = 1
                    code_pseudos_save[pos][i] = pseudodistances.epochs[i][sat][freq]["code"]
                    phase_pseudos_save[pos][i] = pseudodistances.epochs[i][sat][freq]["phase"]
                    iono_error_save[pos][i] = iono_error
                    tropo_error_save[pos][i] = tropo_error
                    ephemeris_error_save[pos][i] = ephemeris_errors[sat]
                    receiver_clock_error_save[pos][i] = receiver_clock_errors[freq]
                    satellite_clock_error_save[pos][i] = sat_pos["clock"]
                    code_multipath_error_save[pos][i] = multipath_code_errors[sat][freq]
                    phase_multipath_error_save[pos][i] = multipath_phase_errors[sat][freq]
                    code_thermal_noise_error_save[pos][i] = thermal_noise_code_errors[sat][freq]
                    phase_thermal_noise_error_save[pos][i] = thermal_noise_phase_errors[sat][freq]
                    sagnac_error_save[pos][i] = sagnac
                    shapiro_error_save[pos][i] = shapiro
                    wind_up_error_save[pos][i] = wind_up[sat][freq]
                    elevation, azimuth = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])
                    elevations[pos][i] = elevation
                    azimuths[pos][i] = azimuth

        visible_satellites_at_epoch = []
        for sat, _ in euclidian_distances.items():
            visible_satellites_at_epoch.append(sat)
        for sat, _ in satellite_positions.epochs[0].items():
            if sat in visible_satellites_at_epoch:
                availability[sat] += 1
            else:
                availability[sat] = 0

    if DISPLAY_RESULTS:
        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], code_pseudos_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Code pseudodistances of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("pseudoranges (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], phase_pseudos_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Phase pseudodistances of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("pseudoranges (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], iono_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Ionospheric errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], tropo_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Tropospheric errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], ephemeris_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Ephemeris errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        prn_to_display = 4
        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j] and j == prn_to_display:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], receiver_clock_error_save[j], "-k", linewidth=2)
        plt.title("Receiver clock error")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], satellite_clock_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Satellite clock errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], code_multipath_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Code multipath errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], phase_multipath_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Phase multipath errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], code_thermal_noise_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Code thermal noise errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], phase_thermal_noise_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Phase thermal noise errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], sagnac_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Sagnac effect errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], shapiro_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Shapiro effect errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        plt.show()

        for j in range(36): #only GPS L1 satellites
            if visible_satellites[j]:
                sat = save_sat(j)
                plt.plot(rx_epochs - rx_epochs[0], wind_up_error_save[j], linewidth=2, label=sat)
        plt.legend(ncol=1, loc="upper right")
        plt.title("Wind-up errors of the GPS L1 visible satellites")
        plt.xlabel("time (s)")
        plt.ylabel("error (m)")
        #plt.savefig("wind_up_testsave", dpi=500)
        plt.show()

        # # plots the apparent position of the satellites in the sky
        # reference_azimuths = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        # reference_elevations = [0, 15, 30, 45, 60, 75]
        # nb_points_reference = 360
        # nb_comet = int(1200 / (rx_epochs[1] - rx_epochs[0])) + 2
        # plt.figure()
        #
        # for epoch in range(2, pseudodistances.nb_epochs):
        #     plt.clf()
        #     for el in reference_elevations:
        #         radius = ((90 - el) / 90) ** 1.05
        #         x = np.linspace(-radius, radius, nb_points_reference)
        #         y = np.sqrt(radius ** 2 - np.power(x, 2))
        #         plt.plot(x, y, "--k", alpha=0.5, linewidth=0.5)
        #         plt.plot(x, -y, "--k", alpha=0.5, linewidth=0.5)
        #         if el:
        #             plt.text(0.02, radius + 0.02, str(el) + "°")
        #     for az in reference_azimuths:
        #         theta = (90 - az) * DEG2RAD
        #         plt.plot([0, np.cos(theta)], [0, np.sin(theta)], "--k", alpha=0.5, linewidth=0.5)
        #         plt.text(np.cos(theta) * 1.05, np.sin(theta) * 1.05, str(az) + "°", rotation=theta * RAD2DEG - 90,
        #                  ha="center", va="center")
        #     for j in range(72 * (GPS + GAL + GLO + BDS)):
        #         x = []
        #         y = []
        #         sat = save_sat(j)
        #         if sat[0] == "G":
        #             col = "-b"
        #         elif sat[0] == "E":
        #             col = "-r"
        #         elif sat[0] == "R":
        #             col = "-g"
        #         else:
        #             col = "orange"
        #         for k in range(min(epoch + 1, nb_comet)):
        #             if azimuths[j][epoch - k] is not None:
        #                 x.append(np.cos((np.pi/2 - azimuths[j][epoch - k])) *
        #                          ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
        #                 y.append(np.sin((np.pi/2 - azimuths[j][epoch - k])) *
        #                          ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
        #                 if len(x) > 1:
        #                     plt.plot([x[-2], x[-1]], [y[-2], y[-1]], col, linewidth=(1 - k/nb_comet)*3)
        #         if len(x):
        #             plt.text(x[0], y[0], sat[0:4], fontweight="semibold")
        #     plt.xlim([-1.3, 1.3])
        #     plt.axis("equal")
        #     plt.xticks([], [])
        #     plt.yticks([], [])
        #     #mng = plt.get_current_fig_manager() # to uncomment to plot full screen
        #     #mng.window.state('zoomed') # to uncomment to plot full screen
        #     plt.pause(0.01)
        # plt.show()

    return pseudodistances


def generate_post_SBAS_correction_pseudodistances(user_positions, rx_epochs, satellite_positions,
                                                  sampling_frequency=GENERATED_SAMPLING_FREQUENCY, label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    The pseudoranges are distorted according to DO229E residual SBAS errors (ref : DO229E 2.5.10.3.1., December 2016)

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""
    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.epoch_start = satellite_positions.epoch_start
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label
    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    # generation of the satellite hardware code offsets
    # no residual error after SBAS correction

    # generation of the satellite hardware phase offsets
    # no residual error after SBAS correction

    # generation of the receiver hardware code offsets
    # no residual error after SBAS correction

    # generation of the receiver hardware phase offsets
    # no residual error after SBAS correction

    # generation of the receiver clock offsets
    # no residual error after SBAS correction

    # generation of the ephemeris error
    # no residual error after SBAS correction

    # generation of the carrier integer error
    # no residual error after SBAS correction

    # generation of the multipath phase error
    multipath_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "E":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "R":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "C":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0

    # generation of the thermal noise code error
    # no residual error after SBAS correction

    # generation of the thermal noise phase error
    # no residual error after SBAS correction

    # generation of the delta tropospheric zenith wet error
    tropo_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat != "time":
            tropo_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                tropo_errors[sat][freq] = 0

    # initializaion of the wind_up error container:
    # no residual error after SBAS correction

    # generation of the UIRE ionospheric delay estimation
    uire_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "E":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "R":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "C":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0

    # generation of the durations of availablilty of the satellites
    availability = {}
    visible_satellites = [0 for _ in range(72 * (GPS + GAL + GLO + BDS))]
    uire_errors_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
    for sat, _ in satellite_positions.epochs[0].items():
        availability[sat] = 0

    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance(user_positions[i], satellite_positions, epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        if i:
            delta_epoch = rx_epochs[i] - rx_epochs[i - 1]
            # generation of the new receiver clock error:
            # no residual error after SBAS correction
            pass

        for sat, sat_pos in euclidian_distances.items():
            elevation = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])[0]
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if EPHEMERIS_ERROR and availability[sat]:
                # generation of the new ephemeris error:
                # no residual error after SBAS correction
                pass

            if TROPOSPHERIC_ERROR and availability[sat]:
                # generation of the new delta tropospheric vertical wet error:
                # no residual error after SBAS correction
                pass

            for freq in frequencies:

                if availability[sat]:

                    # generation of the new multipath code error:
                    multipath_code_errors[sat][freq] = 0.49 + 0.53 * np.exp(-elevation / (10 * DEG2RAD))
                    # generation of the new thermal noise errors:
                    if THERMAL_NOISE:
                        # generation of the new thermal noise code errors:
                        # no residual error after SBAS correction
                        # generation of the new thermal noise phase error:
                        # no residual error after SBAS correction
                        pass

                    # generation of the new UIRE code error:
                    slang_factor = np.power(1 - np.power(EARTH_RADIUS * np.cos(elevation) / (EARTH_RADIUS + IONO_HEIGHT), 2), -0.5)
                    uire_code_std = np.sqrt((0.432 * slang_factor) ** 2 * (1 - np.exp(-2 *
                                    delta_epoch / 120)))
                    uire_errors[sat][freq] = np.exp(-delta_epoch /
                                    120) * uire_errors[sat][freq] + rdm.normal(0, uire_code_std)
                if SAGNAC_EFFECT:
                    # determination of the Sagnac error
                    # no residual error after SBAS correction
                    pass
                if CARRIER_WIND_UP:
                    # determination of the carrier phase wind-up error :
                    # no residual error after SBAS correction
                    pass
                if SHAPIRO_EFFECT:
                    # determination of the Shapiro error
                    # no residual error after SBAS correction
                    pass
                if IONOSPHERIC_ERROR:
                    # determination of the ionospheric error
                    # no residual error after SBAS correction
                    pass
                # tropospheric errors
                tropo_errors[sat][freq] = 0.12 * 1.001 / np.sqrt(0.002001 + np.sin(elevation) ** 2)

                # fast and long term corrections
                error_flt = 0.562

                # generation of the new pseudodistances for the given satellite, frequency and epoch
                sbas_error = rdm.normal(0, np.sqrt(error_flt ** 2 + uire_errors[sat][freq] ** 2
                        + tropo_errors[sat][freq] ** 2 + multipath_code_errors[sat][freq] ** 2))

                # Doppler figure computation :
                if i:
                    user_velocity = (user_positions[i] - user_positions[i - 1]) * sampling_frequency
                else:
                    user_velocity = (user_positions[i + 1] - user_positions[i]) * sampling_frequency
                frequency = find_frequency_from_band(freq)
                doppler = frequency * np.dot(user_velocity - sat_pos["vel"],
                                             (sat_pos["pos"] - user_positions[i]) / sat_pos["distance"]) / C

                # C/N0 estimation
                cn0 = 113 - sat_pos["distance"] * 3.018e-6

                sbas_error = 0

                pseudodistances.epochs[i][sat][freq] = {"code": sat_pos["distance"] + sbas_error,
                        "phase": sat_pos["distance"] + sbas_error / 1500,
                        "doppler": doppler, "CN0": cn0}
                pseudodistances.nb_pseudoranges += 1
                pos = save_pos(sat, freq)
                visible_satellites[pos] = 1
                uire_errors_save[pos][i] = uire_errors[sat][freq]

        visible_satellites_at_epoch = []
        for sat, _ in euclidian_distances.items():
            visible_satellites_at_epoch.append(sat)
        for sat, _ in satellite_positions.epochs[0].items():
            if sat in visible_satellites_at_epoch:
                availability[sat] += 1
            else:
                availability[sat] = 0

    # for j in range(36): #only GPS L1 satellites
    #     if visible_satellites[j]:
    #         sat = save_sat(j)
    #         plt.plot(rx_epochs - rx_epochs[0], uire_errors_save[j], linewidth=2, label=sat)
    # plt.legend(ncol=1, loc="upper right")
    # plt.title("UIRE errors of the GPS L1 visible satellites")
    # plt.xlabel("time (s)")
    # plt.ylabel("error (m)")
    #plt.show()

    return pseudodistances


def generate_post_SBAS_correction_pseudodistances_from_almanach(user_positions, rx_epochs, satellite_positions,
                                                  sampling_frequency=GENERATED_SAMPLING_FREQUENCY, label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    The pseudoranges are distorted according to DO229E residual SBAS errors (ref : DO229E 2.5.10.3.1., December 2016)

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""
    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label
    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    # generation of the satellite hardware code offsets
    # no residual error after SBAS correction

    # generation of the satellite hardware phase offsets
    # no residual error after SBAS correction

    # generation of the receiver hardware code offsets
    # no residual error after SBAS correction

    # generation of the receiver hardware phase offsets
    # no residual error after SBAS correction

    # generation of the receiver clock offsets
    # no residual error after SBAS correction

    # generation of the ephemeris error
    # no residual error after SBAS correction

    # generation of the carrier integer error
    # no residual error after SBAS correction

    # generation of the multipath phase error
    multipath_code_errors = {}
    for sat, _ in satellite_positions.items():
        if sat[0] == "G":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "E":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "R":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "C":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0

    # generation of the thermal noise code error
    # no residual error after SBAS correction

    # generation of the thermal noise phase error
    # no residual error after SBAS correction

    # generation of the delta tropospheric zenith wet error
    tropo_errors = {}
    for sat, _ in satellite_positions.items():
        if sat != "time":
            tropo_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                tropo_errors[sat][freq] = 0

    # initializaion of the wind_up error container:
    # no residual error after SBAS correction

    # generation of the UIRE ionospheric delay estimation
    uire_errors = {}
    for sat, _ in satellite_positions.items():
        if sat[0] == "G":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "E":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "R":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "C":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0

    # generation of the durations of availablilty of the satellites
    availability = {}
    visible_satellites = [0 for _ in range(72 * (GPS + GAL + GLO + BDS))]
    uire_errors_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
    for sat, _ in satellite_positions.items():
        availability[sat] = 0

    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance_from_almanach(user_positions[i], satellite_positions,
                                                               epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        if i:
            delta_epoch = rx_epochs[i] - rx_epochs[i - 1]
            # generation of the new receiver clock error:
            # no residual error after SBAS correction
            pass

        for sat, sat_pos in euclidian_distances.items():
            elevation = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])[0]
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if EPHEMERIS_ERROR and availability[sat]:
                # generation of the new ephemeris error:
                # no residual error after SBAS correction
                pass

            if TROPOSPHERIC_ERROR and availability[sat]:
                # generation of the new delta tropospheric vertical wet error:
                # no residual error after SBAS correction
                pass

            for freq in frequencies:

                if availability[sat]:

                    # generation of the new multipath code error:
                    multipath_code_errors[sat][freq] = 0.49 + 0.53 * np.exp(-elevation / (10 * DEG2RAD))
                    # generation of the new thermal noise errors:
                    if THERMAL_NOISE:
                        # generation of the new thermal noise code errors:
                        # no residual error after SBAS correction
                        # generation of the new thermal noise phase error:
                        # no residual error after SBAS correction
                        pass

                    # generation of the new UIRE code error:
                    slang_factor = np.power(1 - np.power(EARTH_RADIUS * np.cos(elevation) / (EARTH_RADIUS + IONO_HEIGHT), 2), -0.5)
                    uire_code_std = np.sqrt((0.432 * slang_factor) ** 2 * (1 - np.exp(-2 *
                                    delta_epoch / 120)))
                    uire_errors[sat][freq] = np.exp(-delta_epoch /
                                    120) * uire_errors[sat][freq] + rdm.normal(0, uire_code_std)
                if SAGNAC_EFFECT:
                    # determination of the Sagnac error
                    # no residual error after SBAS correction
                    pass
                if CARRIER_WIND_UP:
                    # determination of the carrier phase wind-up error :
                    # no residual error after SBAS correction
                    pass
                if SHAPIRO_EFFECT:
                    # determination of the Shapiro error
                    # no residual error after SBAS correction
                    pass
                if IONOSPHERIC_ERROR:
                    # determination of the ionospheric error
                    # no residual error after SBAS correction
                    pass
                # tropospheric errors
                tropo_errors[sat][freq] = 0.12 * 1.001 / np.sqrt(0.002001 + np.sin(elevation) ** 2)

                # fast and long term corrections
                error_flt = 0.562

                # generation of the new pseudodistances for the given satellite, frequency and epoch
                sbas_error = rdm.normal(0, np.sqrt(error_flt ** 2 + uire_errors[sat][freq] ** 2
                        + tropo_errors[sat][freq] ** 2 + multipath_code_errors[sat][freq] ** 2))

                # Doppler figure computation :
                if i:
                    user_velocity = (user_positions[i] - user_positions[i - 1]) * sampling_frequency
                else:
                    user_velocity = (user_positions[i + 1] - user_positions[i]) * sampling_frequency
                frequency = find_frequency_from_band(freq)
                doppler = frequency * np.dot(user_velocity - sat_pos["vel"],
                                             (sat_pos["pos"] - user_positions[i]) / sat_pos["distance"]) / C

                # C/N0 estimation
                cn0 = 113 - sat_pos["distance"] * 3.018e-6

                sbas_error = 0

                pseudodistances.epochs[i][sat][freq] = {"code": sat_pos["distance"] + sbas_error,
                        "phase": sat_pos["distance"] + sbas_error / 1500,
                        "doppler": doppler, "CN0": cn0}
                pseudodistances.nb_pseudoranges += 1
                pos = save_pos(sat, freq)
                visible_satellites[pos] = 1
                uire_errors_save[pos][i] = uire_errors[sat][freq]

        visible_satellites_at_epoch = []
        for sat, _ in euclidian_distances.items():
            visible_satellites_at_epoch.append(sat)
        for sat, _ in satellite_positions.items():
            if sat in visible_satellites_at_epoch:
                availability[sat] += 1
            else:
                availability[sat] = 0

    # for j in range(36): #only GPS L1 satellites
    #     if visible_satellites[j]:
    #         sat = save_sat(j)
    #         plt.plot(rx_epochs - rx_epochs[0], uire_errors_save[j], linewidth=2, label=sat)
    # plt.legend(ncol=1, loc="upper right")
    # plt.title("UIRE errors of the GPS L1 visible satellites")
    # plt.xlabel("time (s)")
    # plt.ylabel("error (m)")
    #plt.show()

    return pseudodistances


def generate_post_SBAS_correction_pseudodistances_with_meaconer(user_positions, rx_epochs, satellite_positions,
                    meaconer, sampling_frequency=GENERATED_SAMPLING_FREQUENCY, label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    The pseudoranges are distorted according to DO229E residual SBAS errors (ref : DO229E 2.5.10.3.1., December 2016)

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""
    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.epoch_start = satellite_positions.epoch_start
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label
    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    # generation of the satellite hardware code offsets
    # no residual error after SBAS correction

    # generation of the satellite hardware phase offsets
    # no residual error after SBAS correction

    # generation of the receiver hardware code offsets
    # no residual error after SBAS correction

    # generation of the receiver hardware phase offsets
    # no residual error after SBAS correction

    # generation of the receiver clock offsets
    # no residual error after SBAS correction

    # generation of the ephemeris error
    # no residual error after SBAS correction

    # generation of the carrier integer error
    # no residual error after SBAS correction

    # generation of the multipath phase error
    multipath_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "E":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "R":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "C":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0

    # generation of the thermal noise code error
    # no residual error after SBAS correction

    # generation of the thermal noise phase error
    # no residual error after SBAS correction

    # generation of the delta tropospheric zenith wet error
    tropo_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat != "time":
            tropo_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                tropo_errors[sat][freq] = 0

    # initializaion of the wind_up error container:
    # no residual error after SBAS correction

    # generation of the UIRE ionospheric delay estimation
    uire_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "E":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "R":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "C":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0

    # initiallization of the delta_tau of the tracking loops
    previous_delta_tau  = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat != "time":
            previous_delta_tau[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                previous_delta_tau[sat][freq] = None


    # generation of the durations of availablilty of the satellites
    availability = {}
    visible_satellites = [0 for _ in range(72 * (GPS + GAL + GLO + BDS))]
    uire_errors_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
    for sat, _ in satellite_positions.epochs[0].items():
        availability[sat] = 0

    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance(user_positions[i], satellite_positions, epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        if i:
            delta_epoch = rx_epochs[i] - rx_epochs[i - 1]
            # generation of the new receiver clock error:
            # no residual error after SBAS correction
            pass

        for sat, sat_pos in euclidian_distances.items():
            elevation = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])[0]
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if EPHEMERIS_ERROR and availability[sat]:
                # generation of the new ephemeris error:
                # no residual error after SBAS correction
                pass

            if TROPOSPHERIC_ERROR and availability[sat]:
                # generation of the new delta tropospheric vertical wet error:
                # no residual error after SBAS correction
                pass

            for freq in frequencies:

                if availability[sat]:

                    # generation of the new multipath code error:
                    multipath_code_errors[sat][freq] = 0.49 + 0.53 * np.exp(-elevation / (10 * DEG2RAD))
                    # generation of the new thermal noise errors:
                    if THERMAL_NOISE:
                        # generation of the new thermal noise code errors:
                        # no residual error after SBAS correction
                        # generation of the new thermal noise phase error:
                        # no residual error after SBAS correction
                        pass

                    # generation of the new UIRE code error:
                    slang_factor = np.power(1 - np.power(EARTH_RADIUS * np.cos(elevation) / (EARTH_RADIUS + IONO_HEIGHT), 2), -0.5)
                    uire_code_std = np.sqrt((0.432 * slang_factor) ** 2 * (1 - np.exp(-2 *
                                    delta_epoch / 120)))
                    uire_errors[sat][freq] = np.exp(-delta_epoch /
                                    120) * uire_errors[sat][freq] + rdm.normal(0, uire_code_std)
                if SAGNAC_EFFECT:
                    # determination of the Sagnac error
                    # no residual error after SBAS correction
                    pass
                if CARRIER_WIND_UP:
                    # determination of the carrier phase wind-up error :
                    # no residual error after SBAS correction
                    pass
                if SHAPIRO_EFFECT:
                    # determination of the Shapiro error
                    # no residual error after SBAS correction
                    pass
                if IONOSPHERIC_ERROR:
                    # determination of the ionospheric error
                    # no residual error after SBAS correction
                    pass
                # tropospheric errors
                tropo_errors[sat][freq] = 0.12 * 1.001 / np.sqrt(0.002001 + np.sin(elevation) ** 2)

                # fast and long term corrections
                error_flt = 0.562

                # SBAS residual error
                sbas_error = rdm.normal(0, np.sqrt(error_flt ** 2 + uire_errors[sat][freq] ** 2
                                                   + tropo_errors[sat][freq] ** 2 + multipath_code_errors[sat][
                                                       freq] ** 2))

                # Doppler figure computation :
                if i:
                    user_velocity = (user_positions[i] - user_positions[i - 1]) * sampling_frequency
                else:
                    user_velocity = (user_positions[i + 1] - user_positions[i]) * sampling_frequency
                frequency = find_frequency_from_band(freq)
                doppler = frequency * np.dot(user_velocity - sat_pos["vel"],
                                             (sat_pos["pos"] - user_positions[i]) / sat_pos["distance"]) / C

                # C/N0 estimation
                cn0 = 113 - sat_pos["distance"] * 3.018e-6

                # addition of the meaconer's induced bias
                delta_tau = np.linalg.norm(sat_pos["pos"] - np.squeeze(np.transpose(meaconer.ecef_position))) / C - \
                             sat_pos["distance"] / C + meaconer.delay * 1e9 + \
                             np.linalg.norm(user_positions[i] - np.squeeze(np.transpose(meaconer.ecef_position))) / C
                frequency = find_frequency_from_band(freq)
                lamda = C / frequency
                delta_theta = np.mod(delta_tau, lamda) * 2 * np.pi / lamda

                elevation_meaconer = conversion.ECEF2elevation_azimuth(user_positions[i], np.squeeze(np.transpose(meaconer.ecef_position)))[0]
                signal_attenuation = (lamda / 4 / np.pi / np.linalg.norm(user_positions[i] - np.squeeze(np.transpose(meaconer.ecef_position)))) ** 2 \
                                     * maximum_antenna_gain(elevation_meaconer)
                integration_time = 0.02  # in seconds
                inter_chip_spacing = 1 / 1023000  # in seconds
                bandwidth = 5e6  # in Hertz, bandwidth of the front-end filter
                chip_period = 1 / 1023000   # inverse of the chipping rate
                autocorr_function = lambda tau: meaconer_bias_estimation.R_2(tau, bandwidth, chip_period)
                meaconer_bias = meaconer_bias_estimation.compute_meaconer_bias(
                    meaconer_bias_estimation.discriminator_NEMLE, meaconer_bias_estimation.discriminator_atan2,
                    integration_time, doppler + frequency, autocorr_function, inter_chip_spacing,
                    1,
                    [delta_tau],
                    [meaconer.delay * 1e-9],
                    [signal_attenuation],
                    [10 ** (meaconer.gain / 10)], 1, chip_period,
                    previous_delta_tau[sat][freq])
                old_meaconer_bias = meaconer_bias_estimation_old_version.compute_meaconer_bias(
                        delta_tau, meaconer.delay * 1e-9,
                                                                                   signal_attenuation,
                                                                                   10 ** (meaconer.gain / 10),
                                                                                   previous_delta_tau[sat][freq])
                previous_delta_tau[sat][freq] = delta_tau
                # print(meaconer_bias[0] * C)
                # generation of the new pseudodistances for the given satellite, frequency and epoch

                pseudodistances.epochs[i][sat][freq] = {"code": sat_pos["distance"] + sbas_error + meaconer_bias[0] * C,
                        "phase": sat_pos["distance"] + sbas_error / 500 + meaconer_bias[1] * lamda,
                        "doppler": doppler, "CN0": cn0}
                pseudodistances.nb_pseudoranges += 1
                pos = save_pos(sat, freq)
                visible_satellites[pos] = 1
                uire_errors_save[pos][i] = uire_errors[sat][freq]

        visible_satellites_at_epoch = []
        for sat, _ in euclidian_distances.items():
            visible_satellites_at_epoch.append(sat)
        for sat, _ in satellite_positions.epochs[0].items():
            if sat in visible_satellites_at_epoch:
                availability[sat] += 1
            else:
                availability[sat] = 0

    # for j in range(36): #only GPS L1 satellites
    #     if visible_satellites[j]:
    #         sat = save_sat(j)
    #         plt.plot(rx_epochs - rx_epochs[0], uire_errors_save[j], linewidth=2, label=sat)
    # plt.legend(ncol=1, loc="upper right")
    # plt.title("UIRE errors of the GPS L1 visible satellites")
    # plt.xlabel("time (s)")
    # plt.ylabel("error (m)")
    # plt.show()

    return pseudodistances


def generate_post_SBAS_correction_pseudoranges_with_meaconer_in_parallel(user_positions, rx_epochs, satellite_positions,
                    scenarios, grouped_scenarios, sampling_frequency=GENERATED_SAMPLING_FREQUENCY,
                                                                        bar=None, label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    The pseudoranges are distorted according to DO229E residual SBAS errors (ref : DO229E 2.5.10.3.1., December 2016)

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""

    # initialization of the Pseudodistances structure
    for sc in range(len(grouped_scenarios)):
        scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"] = Pseudodistances()
        scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].epoch_start = satellite_positions.epoch_start
        scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].nb_epochs = len(rx_epochs)
        scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].label = label
        if L1:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GPS"].append("L1")
        if L2:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GPS"].append("L2")
        if L5:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GPS"].append("L5")
        if E1:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GAL"].append("E1")
        if E5:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GAL"].append("E5")
        if R1:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GLO"].append("R1")
        if R2:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GLO"].append("R2")
        if B2:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["BDS"].append("B2")
        if B3:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["BDS"].append("B3")

    # generation of the multipath code error
    multipath_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            multipath_code_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GPS"]:
                multipath_code_errors[sat][freq] = rdm.normal(0, 1)
        if sat[0] == "E":
            multipath_code_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GAL"]:
                multipath_code_errors[sat][freq] = rdm.normal(0, 1)
        if sat[0] == "R":
            multipath_code_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GLO"]:
                multipath_code_errors[sat][freq] = rdm.normal(0, 1)
        if sat[0] == "C":
            multipath_code_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["BDS"]:
                multipath_code_errors[sat][freq] = rdm.normal(0, 1)

    # generation of the delta tropospheric zenith wet error
    tropo_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat != "time":
            tropo_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GPS"]:
                tropo_errors[sat][freq] = 0

    # generation of the UIRE ionospheric delay estimation
    uire_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            uire_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GPS"]:
                uire_errors[sat][freq] = rdm.normal(0, 0.432)
        if sat[0] == "E":
            uire_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GAL"]:
                uire_errors[sat][freq] = rdm.normal(0, 0.432)
        if sat[0] == "R":
            uire_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GLO"]:
                uire_errors[sat][freq] = rdm.normal(0, 0.432)
        if sat[0] == "C":
            uire_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["BDS"]:
                uire_errors[sat][freq] = rdm.normal(0, 0.432)

    # initialization of the delta_tau of the tracking loops
    previous_delta_tau = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat != "time":
            previous_delta_tau[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GPS"]:
                previous_delta_tau[sat][freq] = {}
                for sc in range(len(grouped_scenarios)):
                    previous_delta_tau[sat][freq][str(sc)] = None

    # initialization of the meaconer induced bias of the tracking loops
    meaconer_bias_save = [[[None for _ in range(scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].nb_epochs)]
                        for _ in range(36 * (GPS + GAL + GLO + BDS))] for _ in range(len(grouped_scenarios))]
    for sc in range(len(grouped_scenarios)):
        scenarios.spoofed[grouped_scenarios[sc]]["meaconer_biases"] = \
        [[None for _ in range(scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].nb_epochs)]
                        for _ in range(36 * (GPS + GAL + GLO + BDS))]

    # generation of the durations of availablilty of the satellites
    availability = {}
    visible_satellites = [0 for _ in range(72 * (GPS + GAL + GLO + BDS))]
    uire_errors_save = [[None for _ in range(scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].nb_epochs)]
                        for _ in range(72 * (GPS + GAL + GLO + BDS))]
    for sat, _ in satellite_positions.epochs[0].items():
        availability[sat] = 0

    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance(user_positions[i], satellite_positions, epoch, ELEVATION_MASK)
        for sc in grouped_scenarios:
            scenarios.spoofed[sc]["pseudoranges"].epochs.append({"time": epoch})

        if i:
            delta_epoch = rx_epochs[i] - rx_epochs[i - 1]

        for sat, sat_pos in euclidian_distances.items():
            elevation = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])[0]
            for sc in grouped_scenarios:
                scenarios.spoofed[sc]["pseudoranges"].epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GLO"]
            else:
                frequencies = scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["BDS"]

            for freq in frequencies:

                if availability[sat]:

                    # generation of the new multipath code error:
                    multipath_code_errors[sat][freq] = 0.49 + 0.53 * np.exp(-elevation / (10 * DEG2RAD))

                    # generation of the new UIRE code error:
                    slang_factor = np.power(1 - np.power(EARTH_RADIUS * np.cos(elevation) / (EARTH_RADIUS + IONO_HEIGHT), 2), -0.5)

                    uire_code_std = np.sqrt((0.432 * slang_factor) ** 2 * (1 - np.exp(-2 *
                                    delta_epoch / 120)))
                    uire_errors[sat][freq] = np.exp(-delta_epoch /
                                    120) * uire_errors[sat][freq] + rdm.normal(0, uire_code_std)

                # tropospheric errors
                tropo_errors[sat][freq] = 0.12 * 1.001 / np.sqrt(0.002001 + np.sin(elevation) ** 2)

                # fast and long term corrections
                error_flt = 0.562

                # SBAS residual error
                sbas_error = rdm.normal(0, np.sqrt(error_flt ** 2 + uire_errors[sat][freq] ** 2
                                                   + tropo_errors[sat][freq] ** 2 + multipath_code_errors[sat][
                                                       freq] ** 2))

                # Doppler figure computation :
                if i:
                    user_velocity = (user_positions[i] - user_positions[i - 1]) * sampling_frequency
                else:
                    user_velocity = (user_positions[i + 1] - user_positions[i]) * sampling_frequency
                frequency = find_frequency_from_band(freq)
                doppler = frequency * np.dot(user_velocity - sat_pos["vel"],
                                             (sat_pos["pos"] - user_positions[i]) / sat_pos["distance"]) / C

                # C/N0 estimation
                cn0 = 113 - sat_pos["distance"] * 3.018e-6

                for sc in range(len(grouped_scenarios)):

                    # addition of the meaconer's induced bias
                    meaconer = scenarios.spoofed[grouped_scenarios[sc]]["meaconer"]

                    # # simulating the meaconer INSIDE the aircraft
                    # meaconer.ecef_position = [user_positions[i][0] + 0.1,
                    #                           user_positions[i][1] + 0,
                    #                           user_positions[i][2] + 0]

                    # computation of the GNSS signal delay of the spoofed signal (in seconds)
                    delta_tau = np.linalg.norm(sat_pos["pos"] - np.squeeze(np.transpose(meaconer.ecef_position))) / C -\
                                 sat_pos["distance"] / C + meaconer.delay * 1e-9 + \
                                 np.linalg.norm(user_positions[i] - np.squeeze(np.transpose(meaconer.ecef_position))) / C
                    frequency = find_frequency_from_band(freq)
                    lamda = C / frequency
                    delta_theta = np.mod(delta_tau, lamda) * 2 * np.pi / lamda
                    # computation of the free space loss (expressed in linear form)
                    elevation_meaconer = conversion.ECEF2elevation_azimuth(user_positions[i],
                                         np.squeeze(np.transpose(meaconer.ecef_position)))[0]
                    signal_attenuation = (lamda / 4 / np.pi / np.linalg.norm(user_positions[i] -
                                         np.squeeze(np.transpose(meaconer.ecef_position)))) ** 2 \
                                         * maximum_antenna_gain(elevation_meaconer)
                    autocorrelation_function = lambda tau: meaconer_bias_estimation.R_2(tau, BANDWIDTH)

                    meaconer_bias = meaconer_bias_estimation.compute_meaconer_bias(
                       meaconer_bias_estimation.discriminator_NEMLP, meaconer_bias_estimation.discriminator_atan2,
                       INTEGRATION_TIME, doppler + frequency, autocorrelation_function, INTER_CHIP_SPACING,
                       1, [delta_tau], [meaconer.delay * 1e-9], [signal_attenuation],
                       [10 ** (meaconer.gain / 10)], 1, CHIPPING_PERIOD, previous_delta_tau[sat][freq][str(sc)])

                    # old_meaconer_bias = meaconer_bias_estimation_old_version.compute_meaconer_bias(
                    #     delta_tau, meaconer.delay * 1e-9, signal_attenuation, 10 ** (meaconer.gain / 10),
                    #     previous_delta_tau[sat][freq][str(sc)])

                    sbas_error = 0

                    # if sat not in ['G11']:
                    #     meaconer_bias = [0, 0]

                    previous_delta_tau[sat][freq][str(sc)] = meaconer_bias[0]
                    pos = save_pos(sat, freq)
                    meaconer_bias_save[sc][pos][i] = meaconer_bias[0] * C
                    scenarios.spoofed[grouped_scenarios[sc]]["meaconer_biases"][pos][i] = meaconer_bias[0] * C

                    # generation of the new pseudodistances for the given satellite, frequency and epoch
                    scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].epochs[i][sat][freq] = \
                        {"code": sat_pos["distance"] + sbas_error + meaconer_bias[0] * C,
                            "phase": sat_pos["distance"] + sbas_error / 500 + meaconer_bias[1] * lamda,
                            "doppler": doppler, "CN0": cn0, "meaconer_code_bias": meaconer_bias[0] * C}
                    scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].nb_pseudoranges += 1
                    bar()
                pos = save_pos(sat, freq)
                visible_satellites[pos] = 1
                uire_errors_save[pos][i] = uire_errors[sat][freq]




        visible_satellites_at_epoch = []
        for sat, _ in euclidian_distances.items():
            visible_satellites_at_epoch.append(sat)
        for sat, _ in satellite_positions.epochs[0].items():
            if sat in visible_satellites_at_epoch:
                availability[sat] += 1
            else:
                availability[sat] = 0


def generate_post_SBAS_correction_pseudodistances_AP_responses(flight_profile, satellite_positions,
                                                  sampling_frequency=GENERATED_SAMPLING_FREQUENCY, label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    The pseudoranges are distorted according to DO229E residual SBAS errors (ref : DO229E 2.5.10.3.1., December 2016)

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""

    rx_epochs = []
    for wp in range(flight_profile.nb_waypoints):
        rx_epochs.append(flight_profile.waypoints[wp].time + flight_profile.time_start)

    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.epoch_start = satellite_positions.epoch_start
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label

    # initialization of the AP-driven flight profile
    ap_driven_fp = waypoints.FlightProfile()
    ap_driven_fp.time_start = rx_epochs[0]
    ap_driven_fp.add(flight_profile.waypoints[0])

    # initialization of the trajectory estimation containers
    traj_flight_profile = waypoints.FlightProfile()
    traj_flight_profile.time_start = rx_epochs[0]

    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    number_of_constellations = 0
    freq_list = {}

    if pseudodistances.frequencies["GPS"]:
        for freq in pseudodistances.frequencies["GPS"]:
            freq_list.setdefault(freq, number_of_constellations)
        number_of_constellations += 1
    if pseudodistances.frequencies["GAL"]:
        for freq in pseudodistances.frequencies["GAL"]:
            freq_list.setdefault(freq, number_of_constellations)
        number_of_constellations += 1
    if pseudodistances.frequencies["GLO"]:
        for freq in pseudodistances.frequencies["GLO"]:
            freq_list.setdefault(freq, number_of_constellations)
        number_of_constellations += 1
    if pseudodistances.frequencies["BDS"]:
        for freq in pseudodistances.frequencies["BDS"]:
            freq_list.setdefault(freq, number_of_constellations)
        number_of_constellations += 1

    number_of_states = 3 + number_of_constellations

    number_of_samples = pseudodistances.nb_epochs

    ecef_positions = np.zeros((number_of_samples + 1, 3))

    state_vector = np.zeros((number_of_states, 1))
    state_vector[0:3, :] = [[6378137], [0], [0]]


    # generation of the satellite hardware code offsets
    # no residual error after SBAS correction

    # generation of the satellite hardware phase offsets
    # no residual error after SBAS correction

    # generation of the receiver hardware code offsets
    # no residual error after SBAS correction

    # generation of the receiver hardware phase offsets
    # no residual error after SBAS correction

    # generation of the receiver clock offsets
    # no residual error after SBAS correction

    # generation of the ephemeris error
    # no residual error after SBAS correction

    # generation of the carrier integer error
    # no residual error after SBAS correction

    # generation of the multipath phase error
    multipath_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "E":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "R":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "C":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0

    # generation of the thermal noise code error
    # no residual error after SBAS correction

    # generation of the thermal noise phase error
    # no residual error after SBAS correction

    # generation of the delta tropospheric zenith wet error
    tropo_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat != "time":
            tropo_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                tropo_errors[sat][freq] = 0

    # initializaion of the wind_up error container:
    # no residual error after SBAS correction

    # generation of the UIRE ionospheric delay estimation
    uire_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "E":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "R":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "C":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0

    # generation of the durations of availablilty of the satellites
    availability = {}
    visible_satellites = [0 for _ in range(72 * (GPS + GAL + GLO + BDS))]
    uire_errors_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
    for sat, _ in satellite_positions.epochs[0].items():
        availability[sat] = 0

    for i, epoch in enumerate(rx_epochs):
        user_position = np.array([ap_driven_fp.waypoints[-1].x,
                                  ap_driven_fp.waypoints[-1].y,
                                  ap_driven_fp.waypoints[-1].z])
        euclidian_distances = euclidian_distance(user_position, satellite_positions, epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        if i:
            delta_epoch = rx_epochs[i] - rx_epochs[i - 1]
            # generation of the new receiver clock error:
            # no residual error after SBAS correction
            pass

        for sat, sat_pos in euclidian_distances.items():
            elevation = conversion.ECEF2elevation_azimuth(user_position, sat_pos["pos"])[0]
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if EPHEMERIS_ERROR and availability[sat]:
                # generation of the new ephemeris error:
                # no residual error after SBAS correction
                pass

            if TROPOSPHERIC_ERROR and availability[sat]:
                # generation of the new delta tropospheric vertical wet error:
                # no residual error after SBAS correction
                pass

            for freq in frequencies:

                if availability[sat]:

                    # generation of the new multipath code error:
                    multipath_code_errors[sat][freq] = 0.49 + 0.53 * np.exp(-elevation / (10 * DEG2RAD))
                    # generation of the new thermal noise errors:
                    if THERMAL_NOISE:
                        # generation of the new thermal noise code errors:
                        # no residual error after SBAS correction
                        # generation of the new thermal noise phase error:
                        # no residual error after SBAS correction
                        pass

                    # generation of the new UIRE code error:
                    slang_factor = np.power(1 - np.power(EARTH_RADIUS * np.cos(elevation) / (EARTH_RADIUS + IONO_HEIGHT), 2), -0.5)
                    uire_code_std = np.sqrt((0.432 * slang_factor) ** 2 * (1 - np.exp(-2 *
                                    delta_epoch / 120)))
                    uire_errors[sat][freq] = np.exp(-delta_epoch /
                                    120) * uire_errors[sat][freq] + rdm.normal(0, uire_code_std)
                if SAGNAC_EFFECT:
                    # determination of the Sagnac error
                    # no residual error after SBAS correction
                    pass
                if CARRIER_WIND_UP:
                    # determination of the carrier phase wind-up error :
                    # no residual error after SBAS correction
                    pass
                if SHAPIRO_EFFECT:
                    # determination of the Shapiro error
                    # no residual error after SBAS correction
                    pass
                if IONOSPHERIC_ERROR:
                    # determination of the ionospheric error
                    # no residual error after SBAS correction
                    pass
                # tropospheric errors
                tropo_errors[sat][freq] = 0.12 * 1.001 / np.sqrt(0.002001 + np.sin(elevation) ** 2)

                # fast and long term corrections
                error_flt = 0.562

                # generation of the new pseudodistances for the given satellite, frequency and epoch
                sbas_error = rdm.normal(0, np.sqrt(error_flt ** 2 + uire_errors[sat][freq] ** 2
                        + tropo_errors[sat][freq] ** 2 + multipath_code_errors[sat][freq] ** 2))

                # Doppler figure computation :
                if i:
                    user_velocity = (user_position - previous_user_position) * sampling_frequency
                else:
                    next_user_position = np.array([flight_profile.waypoints[1].x,
                                          flight_profile.waypoints[1].y,
                                          flight_profile.waypoints[1].z])
                    user_velocity = (next_user_position - user_position) * sampling_frequency
                frequency = find_frequency_from_band(freq)
                doppler = frequency * np.dot(user_velocity - sat_pos["vel"],
                                             (sat_pos["pos"] - user_position) / sat_pos["distance"]) / C

                # C/N0 estimation
                cn0 = 113 - sat_pos["distance"] * 3.018e-6

                pseudodistances.epochs[i][sat][freq] = {"code": sat_pos["distance"] + sbas_error,
                        "phase": sat_pos["distance"] + sbas_error / 500,
                        "doppler": doppler, "CN0": cn0}
                pseudodistances.nb_pseudoranges += 1
                pos = save_pos(sat, freq)
                visible_satellites[pos] = 1
                uire_errors_save[pos][i] = uire_errors[sat][freq]

        previous_user_position = user_position

        visible_satellites_at_epoch = []
        for sat, _ in euclidian_distances.items():
            visible_satellites_at_epoch.append(sat)
        for sat, _ in satellite_positions.epochs[0].items():
            if sat in visible_satellites_at_epoch:
                availability[sat] += 1
            else:
                availability[sat] = 0

        # trajectory estimation step

        elevation_mask = -90 if not i else -5

        euclidian_distances = euclidian_distance(np.transpose(state_vector[0:3])[0],
                                                                           satellite_positions,
                                                                           pseudodistances.epochs[i]["time"],
                                                                           elevation_mask)
        error_norm = 1e6
        tries_number = 0
        max_tries = 5 + 20 * np.exp(-i / 3)

        while error_norm > 1e-7 and tries_number < max_tries:

            # observation matrix computation
            Y = []
            H = []
            W = []

            for sat, data_sat in pseudodistances.epochs[i].items():
                if sat != "time":

                    for freq, pseudos in data_sat.items():
                        frequency = find_frequency_from_band(freq)
                        elevation = conversion.ECEF2elevation_azimuth(np.transpose(state_vector[0:3])[0], np.transpose(
                            euclidian_distances[sat]["pos"]))[0]

                        distance = np.linalg.norm(euclidian_distances[sat]["pos"] - np.transpose(state_vector[0:3])[0])
                        estimated_pseudo = distance

                        if CODE_FILTERING_BY_THE_PHASE and i:
                            try:
                                pseudorange = 1 / PHASE_FILTER_ORDER * pseudos["code"] + \
                                              (PHASE_FILTER_ORDER - 1) / PHASE_FILTER_ORDER * (
                                                      pseudodistances.epochs[i - 1][sat][
                                                          "code"] +
                                                      C / frequency * (pseudos["phase"] -
                                                                       pseudodistances.epochs[
                                                                           i - 1][sat][
                                                                           "phase"]))
                            except:
                                pseudorange = pseudos["code"]
                        else:
                            pseudorange = pseudos["code"]

                        Y.append([pseudorange - estimated_pseudo])

                        jacobian_x = (state_vector[0, 0] - euclidian_distances[sat]["pos"][0]) / pseudos["code"]
                        jacobian_y = (state_vector[1, 0] - euclidian_distances[sat]["pos"][1]) / pseudos["code"]
                        jacobian_z = (state_vector[2, 0] - euclidian_distances[sat]["pos"][2]) / pseudos["code"]
                        new_obs_line = [jacobian_x, jacobian_y, jacobian_z] + [0] * number_of_constellations
                        new_obs_line[3 + freq_list[freq]] = 1
                        H.append(new_obs_line)

                        # estimating the sigma pseudo
                        slang_factor = np.power(
                            1 - np.power(EARTH_RADIUS * np.cos(elevation) / (EARTH_RADIUS + IONO_HEIGHT), 2), -0.5)
                        sigma_pseudo_squared = 0.562 ** 2 + (0.432 * slang_factor) ** 2 + \
                                               (0.49 + 0.53 * np.exp(-elevation / (10 * DEG2RAD))) ** 2 + \
                                               (0.12 * 1.001 / np.sqrt(0.002001 + np.sin(elevation) ** 2)) ** 2
                        W.append(1 / sigma_pseudo_squared)

            # adjusting the R matrix to the covariance of the GNSS position
            W = np.diag(W)

            # computation of the WLS estimator
            delta_state_vector = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(H), np.dot(W, H))),
                                                      np.transpose(H)), W), Y)
            cov_state_vector = np.linalg.inv(np.dot(np.transpose(H), np.dot(W, H)))

            state_vector += delta_state_vector

            error_norm = np.linalg.norm(delta_state_vector[0:3, 0])
            tries_number += 1

        # conversion into LLA frame to add the waypoint
        [lat, lon, alt] = conversion.ECEF2LLA(state_vector[0:3, 0])
        alt /= 0.3048
        lat *= RAD2DEG
        lon *= RAD2DEG

        ecef_positions[i, :] = [state_vector[0, 0], state_vector[1, 0], state_vector[2, 0]]

        velocity = (ecef_positions[i, :] - ecef_positions[i - 1, :]) / \
                   (pseudodistances.epochs[i]["time"] - pseudodistances.epochs[i - 1]["time"]) \
            if not i else 131.

        new_wp = waypoints.Waypoint(lon, lat, alt, pseudodistances.epochs[i]["time"] -
                                    pseudodistances.epochs[0]["time"], velocity, 0.,
                                    0, "point {}".format(i), 0, 0, cov_state_vector[0, 0],
                                    cov_state_vector[0, 0], cov_state_vector[0, 0])
        traj_flight_profile.add(new_wp)

        # determination of the AP new waypoint to correct the aircraft estimated trajectory to the desired fp

        order_of_the_ap = 5
        deviation = np.array([0., 0., 0.])
        correction_coeff = 0.5
        if i > order_of_the_ap:
            for wp in range(max(0, i - order_of_the_ap), i):
                deviation += np.array([flight_profile.waypoints[wp].x - traj_flight_profile.waypoints[wp].x,
                                      flight_profile.waypoints[wp].y - traj_flight_profile.waypoints[wp].y,
                                      flight_profile.waypoints[wp].z - traj_flight_profile.waypoints[wp].z])
            deviation *= correction_coeff / min(order_of_the_ap, i)

        if i < len(rx_epochs) - 1:
            [lat, lon, alt] = conversion.ECEF2LLA([flight_profile.waypoints[i + 1].x - deviation[0],
                                               flight_profile.waypoints[i + 1].y - deviation[1],
                                               flight_profile.waypoints[i + 1].z - deviation[2]])
            alt /= 0.3048
            lat *= RAD2DEG
            lon *= RAD2DEG
            new_ap_wp = waypoints.Waypoint(lon, lat, alt, epoch -
                                        pseudodistances.epochs[0]["time"], velocity, 0.,
                                        0, "point {}".format(epoch), 0, 0, cov_state_vector[0, 0],
                                        cov_state_vector[0, 0], cov_state_vector[0, 0])
            ap_driven_fp.add(new_ap_wp)

    return pseudodistances, traj_flight_profile, ap_driven_fp


def generate_post_SBAS_correction_pseudodistances_AP_responses_with_meaconer(flight_profile, satellite_positions,
                        meaconer, sampling_frequency=GENERATED_SAMPLING_FREQUENCY, runway_heading=0, label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    The pseudoranges are distorted according to DO229E residual SBAS errors (ref : DO229E 2.5.10.3.1., December 2016)

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""

    rx_epochs = []
    for wp in range(flight_profile.nb_waypoints):
        rx_epochs.append(flight_profile.waypoints[wp].time + flight_profile.time_start)

    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.epoch_start = satellite_positions.epoch_start
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label

    # initialization of the AP-driven flight profile
    ap_driven_fp = waypoints.FlightProfile()
    ap_driven_fp.time_start = rx_epochs[0]
    ap_driven_fp.add(flight_profile.waypoints[0])

    # initialization of the trajectory estimation containers
    traj_flight_profile = waypoints.FlightProfile()
    traj_flight_profile.time_start = rx_epochs[0]

    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    number_of_constellations = 0
    freq_list = {}

    if pseudodistances.frequencies["GPS"]:
        for freq in pseudodistances.frequencies["GPS"]:
            freq_list.setdefault(freq, number_of_constellations)
        number_of_constellations += 1
    if pseudodistances.frequencies["GAL"]:
        for freq in pseudodistances.frequencies["GAL"]:
            freq_list.setdefault(freq, number_of_constellations)
        number_of_constellations += 1
    if pseudodistances.frequencies["GLO"]:
        for freq in pseudodistances.frequencies["GLO"]:
            freq_list.setdefault(freq, number_of_constellations)
        number_of_constellations += 1
    if pseudodistances.frequencies["BDS"]:
        for freq in pseudodistances.frequencies["BDS"]:
            freq_list.setdefault(freq, number_of_constellations)
        number_of_constellations += 1

    number_of_states = 3 + number_of_constellations

    number_of_samples = pseudodistances.nb_epochs

    ecef_positions = np.zeros((number_of_samples + 1, 3))

    state_vector = np.zeros((number_of_states, 1))
    state_vector[0:3, :] = [[6378137], [0], [0]]
    meaconer_residues_save = []

    # generation of the satellite hardware code offsets
    # no residual error after SBAS correction

    # generation of the satellite hardware phase offsets
    # no residual error after SBAS correction

    # generation of the receiver hardware code offsets
    # no residual error after SBAS correction

    # generation of the receiver hardware phase offsets
    # no residual error after SBAS correction

    # generation of the receiver clock offsets
    # no residual error after SBAS correction

    # generation of the ephemeris error
    # no residual error after SBAS correction

    # generation of the carrier integer error
    # no residual error after SBAS correction

    # generation of the multipath phase error
    multipath_code_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "E":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "R":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0
        if sat[0] == "C":
            multipath_code_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    multipath_code_errors[sat][freq] = rdm.normal(0, 1)
                else:
                    multipath_code_errors[sat][freq] = 0

    # generation of the thermal noise code error
    # no residual error after SBAS correction

    # generation of the thermal noise phase error
    # no residual error after SBAS correction

    # generation of the delta tropospheric zenith wet error
    tropo_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat != "time":
            tropo_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                tropo_errors[sat][freq] = 0

    # initializaion of the wind_up error container:
    # no residual error after SBAS correction

    # generation of the UIRE ionospheric delay estimation
    uire_errors = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat[0] == "G":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "E":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GAL"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "R":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["GLO"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0
        if sat[0] == "C":
            uire_errors[sat] = {}
            for freq in pseudodistances.frequencies["BDS"]:
                if MULTIPATH_ERROR:
                    uire_errors[sat][freq] = rdm.normal(0, 0.432)
                else:
                    uire_errors[sat][freq] = 0

    # initiallization of the delta_tau of the tracking loops
    previous_delta_tau = {}
    for sat, _ in satellite_positions.epochs[0].items():
        if sat != "time":
            previous_delta_tau[sat] = {}
            for freq in pseudodistances.frequencies["GPS"]:
                previous_delta_tau[sat][freq] = None

    # generation of the durations of availablilty of the satellites
    availability = {}
    visible_satellites = [0 for _ in range(72 * (GPS + GAL + GLO + BDS))]
    uire_errors_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(36 * (GPS + GAL + GLO + BDS))]
    meaconer_bias_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(36 * (GPS + GAL + GLO + BDS))]
    meaconer_delay_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(36 * (GPS + GAL + GLO + BDS))]
    meaconer_phase_save = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(36 * (GPS + GAL + GLO + BDS))]
    for sat, _ in satellite_positions.epochs[0].items():
        availability[sat] = 0

    for i, epoch in enumerate(rx_epochs):
        user_position = np.array([ap_driven_fp.waypoints[-1].x,
                                  ap_driven_fp.waypoints[-1].y,
                                  ap_driven_fp.waypoints[-1].z])
        euclidian_distances = euclidian_distance(user_position, satellite_positions, epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        if i:
            delta_epoch = rx_epochs[i] - rx_epochs[i - 1]
            # generation of the new receiver clock error:
            # no residual error after SBAS correction
            pass

        for sat, sat_pos in euclidian_distances.items():
            elevation = conversion.ECEF2elevation_azimuth(user_position, sat_pos["pos"])[0]
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if EPHEMERIS_ERROR and availability[sat]:
                # generation of the new ephemeris error:
                # no residual error after SBAS correction
                pass

            if TROPOSPHERIC_ERROR and availability[sat]:
                # generation of the new delta tropospheric vertical wet error:
                # no residual error after SBAS correction
                pass

            for freq in frequencies:

                if availability[sat]:

                    # generation of the new multipath code error:
                    multipath_code_errors[sat][freq] = 0.49 + 0.53 * np.exp(-elevation / (10 * DEG2RAD))
                    # generation of the new thermal noise errors:
                    if THERMAL_NOISE:
                        # generation of the new thermal noise code errors:
                        # no residual error after SBAS correction
                        # generation of the new thermal noise phase error:
                        # no residual error after SBAS correction
                        pass

                    # generation of the new UIRE code error:
                    slang_factor = np.power(1 - np.power(EARTH_RADIUS * np.cos(elevation) / (EARTH_RADIUS + IONO_HEIGHT), 2), -0.5)
                    uire_code_std = np.sqrt((0.432 * slang_factor) ** 2 * (1 - np.exp(-2 *
                                    delta_epoch / 120)))
                    uire_errors[sat][freq] = np.exp(-delta_epoch /
                                    120) * uire_errors[sat][freq] + rdm.normal(0, uire_code_std)
                if SAGNAC_EFFECT:
                    # determination of the Sagnac error
                    # no residual error after SBAS correction
                    pass
                if CARRIER_WIND_UP:
                    # determination of the carrier phase wind-up error :
                    # no residual error after SBAS correction
                    pass
                if SHAPIRO_EFFECT:
                    # determination of the Shapiro error
                    # no residual error after SBAS correction
                    pass
                if IONOSPHERIC_ERROR:
                    # determination of the ionospheric error
                    # no residual error after SBAS correction
                    pass
                # tropospheric errors
                tropo_errors[sat][freq] = 0.12 * 1.001 / np.sqrt(0.002001 + np.sin(elevation) ** 2)

                # fast and long term corrections
                error_flt = 0.562

                # generation of the new pseudodistances for the given satellite, frequency and epoch
                sbas_error = rdm.normal(0, np.sqrt(error_flt ** 2 + uire_errors[sat][freq] ** 2
                        + tropo_errors[sat][freq] ** 2 + multipath_code_errors[sat][freq] ** 2))

                # Doppler figure computation :
                if i:
                    user_velocity = (user_position - previous_user_position) * sampling_frequency
                else:
                    next_user_position = np.array([flight_profile.waypoints[1].x,
                                          flight_profile.waypoints[1].y,
                                          flight_profile.waypoints[1].z])
                    user_velocity = (next_user_position - user_position) * sampling_frequency
                frequency = find_frequency_from_band(freq)
                doppler = frequency * np.dot(user_velocity - sat_pos["vel"],
                                             (sat_pos["pos"] - user_position) / sat_pos["distance"]) / C

                # C/N0 estimation
                cn0 = 113 - sat_pos["distance"] * 3.018e-6

                # addition of the meaconer's induced bias
                delta_tau = np.linalg.norm(sat_pos["pos"] - np.squeeze(np.transpose(meaconer.ecef_position))) / C - \
                            sat_pos["distance"] / C + meaconer.delay * 1e9 + \
                            np.linalg.norm(user_position - np.squeeze(np.transpose(meaconer.ecef_position))) / C
                frequency = find_frequency_from_band(freq)
                lamda = C / frequency
                delta_theta = np.mod(delta_tau * C, lamda) * 2 * np.pi / lamda
                # computation of the free space loss (expressed in linear form)
                elevation_meaconer = conversion.ECEF2elevation_azimuth(user_position,
                                                                       np.squeeze(
                                                                           np.transpose(meaconer.ecef_position)))[0]
                signal_attenuation = (lamda / 4 / np.pi / np.linalg.norm(user_position -
                                                                         np.squeeze(np.transpose(
                                                                             meaconer.ecef_position)))) ** 2 \
                                     * maximum_antenna_gain(elevation_meaconer)
                autocorrelation_function = lambda tau: meaconer_bias_estimation.R_2(tau, BANDWIDTH)

                meaconer_bias = meaconer_bias_estimation.compute_meaconer_bias(
                    meaconer_bias_estimation.discriminator_NEMLE, meaconer_bias_estimation.discriminator_atan2,
                    INTEGRATION_TIME, doppler + frequency, autocorrelation_function, INTER_CHIP_SPACING,
                    1, [delta_tau], [meaconer.delay * 1e-9], [signal_attenuation],
                    [10 ** (meaconer.gain / 10)], 1, CHIPPING_PERIOD, previous_delta_tau[sat][freq])

                sbas_error = 0

                previous_delta_tau[sat][freq] = meaconer_bias[0]

                pseudodistances.epochs[i][sat][freq] = {"code": sat_pos["distance"] + sbas_error + meaconer_bias[0] * C,
                        "phase": sat_pos["distance"] + sbas_error / 500 + meaconer_bias[1] * lamda,
                        "doppler": doppler, "CN0": cn0}
                pseudodistances.nb_pseudoranges += 1
                pos = save_pos(sat, freq)
                visible_satellites[pos] = 1
                uire_errors_save[pos][i] = uire_errors[sat][freq]
                meaconer_bias_save[pos][i] = meaconer_bias[0] * C
                meaconer_delay_save[pos][i] = delta_tau
                meaconer_phase_save[pos][i] = delta_theta

        previous_user_position = user_position

        visible_satellites_at_epoch = []
        for sat, _ in euclidian_distances.items():
            visible_satellites_at_epoch.append(sat)
        for sat, _ in satellite_positions.epochs[0].items():
            if sat in visible_satellites_at_epoch:
                availability[sat] += 1
            else:
                availability[sat] = 0

        # trajectory estimation step

        elevation_mask = -90 if not i else -5

        euclidian_distances = euclidian_distance(np.transpose(state_vector[0:3])[0],
                                                                           satellite_positions,
                                                                           pseudodistances.epochs[i]["time"],
                                                                           elevation_mask)
        error_norm = 1e6
        tries_number = 0
        max_tries = 5 + 20 * np.exp(-i / 3)

        while error_norm > 1e-7 and tries_number < max_tries:

            # observation matrix computation
            Y = []
            H = []
            W = []
            mu = []

            for sat, data_sat in pseudodistances.epochs[i].items():
                if sat != "time":

                    for freq, pseudos in data_sat.items():
                        frequency = find_frequency_from_band(freq)
                        elevation = conversion.ECEF2elevation_azimuth(np.transpose(state_vector[0:3])[0], np.transpose(
                            euclidian_distances[sat]["pos"]))[0]

                        distance = np.linalg.norm(euclidian_distances[sat]["pos"] - np.transpose(state_vector[0:3])[0])
                        estimated_pseudo = distance

                        if CODE_FILTERING_BY_THE_PHASE and i:
                            try:
                                pseudorange = 1 / PHASE_FILTER_ORDER * pseudos["code"] + \
                                              (PHASE_FILTER_ORDER - 1) / PHASE_FILTER_ORDER * (
                                                      pseudodistances.epochs[i - 1][sat][
                                                          "code"] +
                                                      C / frequency * (pseudos["phase"] -
                                                                       pseudodistances.epochs[
                                                                           i - 1][sat][
                                                                           "phase"]))
                            except:
                                pseudorange = pseudos["code"]
                        else:
                            pseudorange = pseudos["code"]

                        Y.append([pseudorange - estimated_pseudo])

                        jacobian_x = (state_vector[0, 0] - euclidian_distances[sat]["pos"][0]) / pseudos["code"]
                        jacobian_y = (state_vector[1, 0] - euclidian_distances[sat]["pos"][1]) / pseudos["code"]
                        jacobian_z = (state_vector[2, 0] - euclidian_distances[sat]["pos"][2]) / pseudos["code"]
                        new_obs_line = [jacobian_x, jacobian_y, jacobian_z] + [0] * number_of_constellations
                        new_obs_line[3 + freq_list[freq]] = 1
                        H.append(new_obs_line)

                        # estimating the sigma pseudo
                        slang_factor = np.power(
                            1 - np.power(EARTH_RADIUS * np.cos(elevation) / (EARTH_RADIUS + IONO_HEIGHT), 2), -0.5)
                        sigma_pseudo_squared = 0.562 ** 2 + (0.432 * slang_factor) ** 2 + \
                                               (0.49 + 0.53 * np.exp(-elevation / (10 * DEG2RAD))) ** 2 + \
                                               (0.12 * 1.001 / np.sqrt(0.002001 + np.sin(elevation) ** 2)) ** 2
                        W.append(1 / sigma_pseudo_squared)
                        pos = save_pos(sat, freq)
                        mu.append([meaconer_bias_save[pos][i]])

            # adjusting the R matrix to the covariance of the GNSS position
            W = np.diag(W)

            # computation of the WLS estimator
            delta_state_vector = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(H), np.dot(W, H))),
                                                      np.transpose(H)), W), Y)
            cov_state_vector = np.linalg.inv(np.dot(np.transpose(H), np.dot(W, H)))

            state_vector += delta_state_vector

            error_norm = np.linalg.norm(delta_state_vector[0:3, 0])
            tries_number += 1

        # conversion into LLA frame to add the waypoint
        [lat, lon, alt] = conversion.ECEF2LLA(state_vector[0:3, 0])
        alt /= 0.3048
        lat *= RAD2DEG
        lon *= RAD2DEG

        ecef_positions[i, :] = [state_vector[0, 0], state_vector[1, 0], state_vector[2, 0]]

        if i:
            velocity = (ecef_positions[i, :] - ecef_positions[i - 1, :]) / \
                   (pseudodistances.epochs[i]["time"] - pseudodistances.epochs[i - 1]["time"])
            velocity = np.linalg.norm(velocity) * MPS2KT
        else:
            velocity = 131

        new_wp = waypoints.Waypoint(lon, lat, alt, pseudodistances.epochs[i]["time"] -
                                    pseudodistances.epochs[0]["time"], velocity, 0.,
                                    0, "point {}".format(i), 0, 0, cov_state_vector[0, 0],
                                    cov_state_vector[0, 0], cov_state_vector[0, 0])
        traj_flight_profile.add(new_wp)

        # computation of the meaconer induced residue : S * mu

        meaconer_residues = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(H), np.dot(W, H))),
                                                      np.transpose(H)), W), mu)
        meaconer_residues_enu = conversion.ECEF2ENU(meaconer_residues[:3],
                                flight_profile.waypoints[i].lat * DEG2RAD, flight_profile.waypoints[i].lon * DEG2RAD)
        along_track_residue = np.sin(runway_heading) * meaconer_residues_enu[0] + np.cos(runway_heading) * meaconer_residues_enu[1]
        ortho_track_residue = np.cos(runway_heading) * meaconer_residues_enu[0] - np.sin(runway_heading) * meaconer_residues_enu[1]
        meaconer_residues_save.append([along_track_residue, ortho_track_residue, meaconer_residues_enu[2]])

        # determination of the AP new waypoint to correct the aircraft estimated trajectory to the desired fp

        order_of_the_ap = 4
        deviation = np.array([0., 0., 0.])
        correction_coeff = 0
        if i > order_of_the_ap:
            for wp in range(max(0, i - order_of_the_ap), i):
                deviation += np.array([flight_profile.waypoints[wp].x - traj_flight_profile.waypoints[wp].x,
                                      flight_profile.waypoints[wp].y - traj_flight_profile.waypoints[wp].y,
                                      flight_profile.waypoints[wp].z - traj_flight_profile.waypoints[wp].z])
            deviation *= correction_coeff / min(order_of_the_ap, i)

        if i < len(rx_epochs) - 1:
            [lat, lon, alt] = conversion.ECEF2LLA([flight_profile.waypoints[i + 1].x + deviation[0],
                                               flight_profile.waypoints[i + 1].y + deviation[1],
                                               flight_profile.waypoints[i + 1].z + deviation[2]])
            alt /= 0.3048
            lat *= RAD2DEG
            lon *= RAD2DEG
            new_ap_wp = waypoints.Waypoint(lon, lat, alt, epoch -
                                        pseudodistances.epochs[0]["time"] + 1 / sampling_frequency, velocity, 0.,
                                        0, "point {}".format(i + 1), 0, 0, cov_state_vector[0, 0],
                                        cov_state_vector[0, 0], cov_state_vector[0, 0])
            ap_driven_fp.add(new_ap_wp)

    return pseudodistances, meaconer_bias_save, meaconer_residues_save, traj_flight_profile, ap_driven_fp, \
           meaconer_delay_save, meaconer_phase_save


def display_skyplot_and_meaconer_positions(user_positions, rx_epochs, satellite_positions, meaconer_positions=[],
                              label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    Depending on the noise parameters, the pseudodistances are distorted to match realistic behaviours.

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""

    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.epoch_start = satellite_positions.epoch_start
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label
    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    elevations = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
    azimuths = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]

    nb_meaconers = len(meaconer_positions)

    elevations_meaconer = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(nb_meaconers)]
    azimuths_meaconer = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(nb_meaconers)]

    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance(user_positions[i], satellite_positions, epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        for sat, sat_pos in euclidian_distances.items():
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if len(frequencies):
                elevation, azimuth = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])
                pos = save_pos(sat, frequencies[0])
                elevations[pos][i] = elevation
                azimuths[pos][i] = azimuth

        for meaconer in range(nb_meaconers):
            if i < len(meaconer_positions[meaconer]):
                meaconer_position = meaconer_positions[meaconer][i]
            else:
                meaconer_position = meaconer_positions[meaconer][0]
            elevation_meaconer, azimuth_meaconer = conversion.ECEF2elevation_azimuth(user_positions[i],
                                                                                     meaconer_position)
            elevations_meaconer[meaconer][i] = elevation_meaconer
            azimuths_meaconer[meaconer][i] = azimuth_meaconer

    # plots the apparent position of the satellites in the sky
    reference_azimuths = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    reference_elevations = [-15, 0, 15, 30, 45, 60, 75]
    nb_points_reference = 360
    nb_comet = int(100 / (rx_epochs[1] - rx_epochs[0])) + 2
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    epoch = pseudodistances.nb_epochs - 1

    for el in reference_elevations:
        radius = ((90 - el) / 90) ** 1.05
        x = np.linspace(-radius, radius, nb_points_reference)
        y = np.sqrt(radius ** 2 - np.power(x, 2))
        plt.plot(x, y, "--k", alpha=0.5, linewidth=0.5)
        plt.plot(x, -y, "--k", alpha=0.5, linewidth=0.5)
        if el != reference_elevations[0]:
            plt.text(0.02, radius + 0.02, str(el) + "°")
    for az in reference_azimuths:
        theta = (90 - az) * DEG2RAD
        plt.plot([0, np.cos(theta) * 1.2], [0, np.sin(theta) * 1.2], "--k", alpha=0.5, linewidth=0.5)
        plt.text(np.cos(theta) * 1.24, np.sin(theta) * 1.24, str(az) + "°", rotation=theta * RAD2DEG - 90,
                 ha="center", va="center")

    for j in range(72 * (GPS + GAL + GLO + BDS)):
        x = []
        y = []
        sat = save_sat(j)
        if sat[0] == "G":
            col = "-b"
        elif sat[0] == "E":
            col = "-r"
        elif sat[0] == "R":
            col = "-g"
        else:
            col = "orange"
        for k in range(min(epoch + 1, nb_comet)):
            if azimuths[j][epoch - k] is not None:
                x.append(np.cos((np.pi/2 - azimuths[j][epoch - k])) *
                         ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                y.append(np.sin((np.pi/2 - azimuths[j][epoch - k])) *
                         ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                if len(x) > 1:
                    plt.plot([x[-2], x[-1]], [y[-2], y[-1]], col, linewidth=(1 - k/nb_comet)*3.5)
        if len(x):
            plt.text(x[0], y[0], sat[0:4], fontweight="semibold")

    for meaconer in range(nb_meaconers):
        x = []
        y = []

        for k in range(min(epoch + 1, nb_comet)):
            if azimuths_meaconer[meaconer][epoch - k] is not None:
                x.append(np.cos((np.pi/2 - azimuths_meaconer[meaconer][epoch - k])) *
                         ((90 - elevations_meaconer[meaconer][epoch - k] * RAD2DEG) / 90))
                y.append(np.sin((np.pi/2 - azimuths_meaconer[meaconer][epoch - k])) *
                         ((90 - elevations_meaconer[meaconer][epoch - k] * RAD2DEG) / 90))

        if len(x):
            plt.text(x[0], y[0], "M".format(meaconer), fontweight="semibold", va="bottom")
            segments = np.array([np.column_stack([x[i:i + 2],
                                                  y[i:i + 2]])
                                 for i in range(len(y) - 1, 0, -1)])
            lc = LineCollection(segments, cmap='jet', array=np.array(rx_epochs) - rx_epochs[0], linewidth=5)
            ax.add_collection(lc)
            plt.colorbar(lc, label='time (s)', ax=ax)

    plt.xlim([-1.3, 1.3])
    plt.axis("equal")
    plt.xticks([], [])
    plt.yticks([], [])
    #mng = plt.get_current_fig_manager() # to uncomment to plot full screen
    #mng.window.state('zoomed') # to uncomment to plot full screen
    plt.show()


def display_skyplot_and_meaconer_positions_animation(user_positions, rx_epochs, satellite_positions,
                                                     meaconer_positions=[], label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    Depending on the noise parameters, the pseudodistances are distorted to match realistic behaviours.

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""

    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.epoch_start = satellite_positions.epoch_start
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label
    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    elevations = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
    azimuths = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]

    nb_meaconers = len(meaconer_positions)

    elevations_meaconer = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(nb_meaconers)]
    azimuths_meaconer = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(nb_meaconers)]

    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance(user_positions[i], satellite_positions, epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        for sat, sat_pos in euclidian_distances.items():
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if len(frequencies):
                elevation, azimuth = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])
                pos = save_pos(sat, frequencies[0])
                elevations[pos][i] = elevation
                azimuths[pos][i] = azimuth

        for meaconer in range(nb_meaconers):
            if i < len(meaconer_positions[meaconer]):
                meaconer_position = meaconer_positions[meaconer][i]
            else:
                meaconer_position = meaconer_positions[meaconer][0]
            elevation_meaconer, azimuth_meaconer = conversion.ECEF2elevation_azimuth(user_positions[i],
                                                                                     meaconer_position)
            elevations_meaconer[meaconer][i] = elevation_meaconer
            azimuths_meaconer[meaconer][i] = azimuth_meaconer

    # plots the apparent position of the satellites in the sky
    reference_azimuths = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    reference_elevations = [-15, 0, 15, 30, 45, 60, 75]
    nb_points_reference = 360
    nb_comet = int(100 / (rx_epochs[1] - rx_epochs[0])) + 2
    plt.figure()

    for epoch in range(2, pseudodistances.nb_epochs):
        plt.clf()
        for el in reference_elevations:
            radius = ((90 - el) / 90) ** 1.05
            x = np.linspace(-radius, radius, nb_points_reference)
            y = np.sqrt(radius ** 2 - np.power(x, 2))
            plt.plot(x, y, "--k", alpha=0.5, linewidth=0.5)
            plt.plot(x, -y, "--k", alpha=0.5, linewidth=0.5)
            if el != reference_elevations[0]:
                plt.text(0.02, radius + 0.02, str(el) + "°")
        for az in reference_azimuths:
            theta = (90 - az) * DEG2RAD
            plt.plot([0, np.cos(theta) * 1.2], [0, np.sin(theta) * 1.2], "--k", alpha=0.5, linewidth=0.5)
            plt.text(np.cos(theta) * 1.26, np.sin(theta) * 1.26, str(az) + "°", rotation=theta * RAD2DEG - 90,
                     ha="center", va="center")

        for j in range(72 * (GPS + GAL + GLO + BDS)):
            x = []
            y = []
            sat = save_sat(j)
            if sat[0] == "G":
                col = "-b"
            elif sat[0] == "E":
                col = "-r"
            elif sat[0] == "R":
                col = "-g"
            else:
                col = "orange"
            for k in range(min(epoch + 1, nb_comet)):
                if azimuths[j][epoch - k] is not None:
                    x.append(np.cos((np.pi/2 - azimuths[j][epoch - k])) *
                             ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                    y.append(np.sin((np.pi/2 - azimuths[j][epoch - k])) *
                             ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                    if len(x) > 1:
                        plt.plot([x[-2], x[-1]], [y[-2], y[-1]], col, linewidth=(1 - k/nb_comet)*3.5)
            if len(x):
                plt.text(x[0], y[0], sat[0:4], fontweight="semibold")

        for meaconer in range(nb_meaconers):
            x = []
            y = []
            for k in range(min(epoch + 1, nb_comet)):
                if azimuths_meaconer[meaconer][epoch - k] is not None:
                    x.append(np.cos((np.pi/2 - azimuths_meaconer[meaconer][epoch - k])) *
                             ((90 - elevations_meaconer[meaconer][epoch - k] * RAD2DEG) / 90))
                    y.append(np.sin((np.pi/2 - azimuths_meaconer[meaconer][epoch - k])) *
                             ((90 - elevations_meaconer[meaconer][epoch - k] * RAD2DEG) / 90))
                    if len(x) > 1:
                        plt.plot([x[-2], x[-1]], [y[-2], y[-1]], "-r", linewidth=(1 - k/nb_comet)*4)
            if len(x):
                plt.text(x[0], y[0], "M".format(meaconer), fontweight="semibold")

        plt.xlim([-1.3, 1.3])
        plt.axis("equal")
        plt.xticks([], [])
        plt.yticks([], [])
        #mng = plt.get_current_fig_manager() # to uncomment to plot full screen
        #mng.window.state('zoomed') # to uncomment to plot full screen
        plt.pause(0.0006)
    plt.show()


def display_skyplot_and_meaconer_positions_from_almanach_animation(user_positions, rx_epochs, satellite_positions,
                                                     meaconer_positions=[], label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    Depending on the noise parameters, the pseudodistances are distorted to match realistic behaviours.

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""

    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label
    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    elevations = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
    azimuths = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]

    nb_meaconers = len(meaconer_positions)

    elevations_meaconer = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(nb_meaconers)]
    azimuths_meaconer = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(nb_meaconers)]

    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance_from_almanach(user_positions[i], satellite_positions, epoch,
                                                               ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        for sat, sat_pos in euclidian_distances.items():
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if len(frequencies):
                elevation, azimuth = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])
                pos = save_pos(sat, frequencies[0])
                elevations[pos][i] = elevation
                azimuths[pos][i] = azimuth

        for meaconer in range(nb_meaconers):
            if i < len(meaconer_positions[meaconer]):
                meaconer_position = meaconer_positions[meaconer][i]
            else:
                meaconer_position = meaconer_positions[meaconer][0]
            elevation_meaconer, azimuth_meaconer = conversion.ECEF2elevation_azimuth(user_positions[i],
                                                                                     meaconer_position)
            elevations_meaconer[meaconer][i] = elevation_meaconer
            azimuths_meaconer[meaconer][i] = azimuth_meaconer

    # plots the apparent position of the satellites in the sky
    reference_azimuths = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    reference_elevations = [0, 15, 30, 45, 60, 75]
    nb_points_reference = 360
    nb_comet = int(100 / (rx_epochs[1] - rx_epochs[0])) + 2
    plt.figure()

    for epoch in range(2, pseudodistances.nb_epochs):
        plt.clf()
        for el in reference_elevations:
            radius = ((90 - el) / 90) ** 1.05
            x = np.linspace(-radius, radius, nb_points_reference)
            y = np.sqrt(radius ** 2 - np.power(x, 2))
            plt.plot(x, y, "--k", alpha=0.5, linewidth=0.5)
            plt.plot(x, -y, "--k", alpha=0.5, linewidth=0.5)
            if el != reference_elevations[0]:
                plt.text(0.02, radius + 0.02, str(el) + "°")
        for az in reference_azimuths:
            theta = (90 - az) * DEG2RAD
            plt.plot([0, np.cos(theta) * 1], [0, np.sin(theta) * 1], "--k", alpha=0.5, linewidth=0.5)
            plt.text(np.cos(theta) * 1.05, np.sin(theta) * 1.05, str(az) + "°", rotation=theta * RAD2DEG - 90,
                     ha="center", va="center")

        for j in range(72 * (GPS + GAL + GLO + BDS)):
            x = []
            y = []
            sat = save_sat(j)
            if sat[0] == "G":
                col = "-b"
            elif sat[0] == "E":
                col = "-r"
            elif sat[0] == "R":
                col = "-g"
            else:
                col = "orange"
            for k in range(min(epoch + 1, nb_comet)):
                if azimuths[j][epoch - k] is not None:
                    x.append(np.cos((np.pi/2 - azimuths[j][epoch - k])) *
                             ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                    y.append(np.sin((np.pi/2 - azimuths[j][epoch - k])) *
                             ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                    if len(x) > 1:
                        plt.plot([x[-2], x[-1]], [y[-2], y[-1]], col, linewidth=(1 - k/nb_comet)*3.5)
            if len(x):
                plt.text(x[0], y[0], sat[0:4], fontweight="semibold")

        for meaconer in range(nb_meaconers):
            x = []
            y = []
            for k in range(min(epoch + 1, nb_comet)):
                if azimuths_meaconer[meaconer][epoch - k] is not None:
                    x.append(np.cos((np.pi/2 - azimuths_meaconer[meaconer][epoch - k])) *
                             ((90 - elevations_meaconer[meaconer][epoch - k] * RAD2DEG) / 90))
                    y.append(np.sin((np.pi/2 - azimuths_meaconer[meaconer][epoch - k])) *
                             ((90 - elevations_meaconer[meaconer][epoch - k] * RAD2DEG) / 90))
                    if len(x) > 1:
                        plt.plot([x[-2], x[-1]], [y[-2], y[-1]], "-r", linewidth=(1 - k/nb_comet)*4)
            if len(x):
                plt.text(x[0], y[0], "M".format(meaconer), fontweight="semibold")

        plt.xlim([-1.3, 1.3])
        plt.axis("equal")
        plt.xticks([], [])
        plt.yticks([], [])
        # mng = plt.get_current_fig_manager() # to uncomment to plot full screen
        # mng.window.state('zoomed') # to uncomment to plot full screen
        plt.pause(0.5)
    plt.show()


def display_skyplot_and_meaconer_positions_animation_with_color_code(user_positions, rx_epochs, satellite_positions,
                                                     meaconer_positions=[], meaconer_biases=[], label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    Depending on the noise parameters, the pseudodistances are distorted to match realistic behaviours.

    INPUTS :
        user_positions : the nb_epochs x 3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""

    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label
    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    elevations = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
    azimuths = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]

    nb_meaconers = len(meaconer_positions)

    elevations_meaconer = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(nb_meaconers)]
    azimuths_meaconer = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(nb_meaconers)]

    colors = ['#069002', '#05BD00', '#5FDE01', '#B3EC10', '#EEF500', '#F5C800', '#F59C00', '#F56800', '#F53B00', '#F50000']
    color_thresholds = [0.8, 1.5, 3, 5, 8, 13, 21, 34, 55, 100]
    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance(user_positions[i], satellite_positions, epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        for sat, sat_pos in euclidian_distances.items():
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if len(frequencies):
                elevation, azimuth = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])
                pos = save_pos(sat, frequencies[0])
                elevations[pos][i] = elevation
                azimuths[pos][i] = azimuth

        for meaconer in range(nb_meaconers):
            if i < len(meaconer_positions[meaconer]):
                meaconer_position = meaconer_positions[meaconer][i]
            else:
                meaconer_position = meaconer_positions[meaconer][0]
            elevation_meaconer, azimuth_meaconer = conversion.ECEF2elevation_azimuth(user_positions[i],
                                                                                     meaconer_position)
            elevations_meaconer[meaconer][i] = elevation_meaconer
            azimuths_meaconer[meaconer][i] = azimuth_meaconer

    # plots the apparent position of the satellites in the sky
    reference_azimuths = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    reference_elevations = [-15, 0, 15, 30, 45, 60, 75]
    nb_points_reference = 360
    nb_comet = int(1 / (rx_epochs[1] - rx_epochs[0])) + 2
    nb_comet_meaconer = int(100 / (rx_epochs[1] - rx_epochs[0])) + 2
    plt.subplots(1, 1, figsize=(7, 5))

    for epoch in range(2, pseudodistances.nb_epochs):
        plt.clf()
        for el in reference_elevations:
            radius = ((90 - el) / 90) ** 1.05
            x = np.linspace(-radius, radius, nb_points_reference)
            y = np.sqrt(radius ** 2 - np.power(x, 2))
            plt.plot(x, y, "--k", alpha=0.5, linewidth=0.5)
            plt.plot(x, -y, "--k", alpha=0.5, linewidth=0.5)
            if el != reference_elevations[0]:
                plt.text(0.02, radius + 0.02, str(el) + "°")
        for az in reference_azimuths:
            theta = (90 - az) * DEG2RAD
            plt.plot([0, np.cos(theta) * 1.2], [0, np.sin(theta) * 1.2], "--k", alpha=0.5, linewidth=0.5)
            plt.text(np.cos(theta) * 1.24, np.sin(theta) * 1.24, str(az) + "°", rotation=theta * RAD2DEG - 90,
                     ha="center", va="center")

        for j in range(36 * (GPS + GAL + GLO + BDS)):
            x = []
            y = []
            sat = save_sat(j)
            if sat[0] == "G":
                col = "-b"
            elif sat[0] == "E":
                col = "-r"
            elif sat[0] == "R":
                col = "-g"
            else:
                col = "orange"
            for k in range(min(epoch + 1, nb_comet)):
                if azimuths[j][epoch - k] is not None:
                    x.append(np.cos((np.pi/2 - azimuths[j][epoch - k])) *
                             ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                    y.append(np.sin((np.pi/2 - azimuths[j][epoch - k])) *
                             ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                    if len(x) > 1:
                        plt.plot([x[-2], x[-1]], [y[-2], y[-1]], col, linewidth=(1 - k/nb_comet)*3.5)
            if len(x):
                plt.text(x[0], y[0], sat[0:4], fontweight="semibold")

            if meaconer_biases[j][epoch] is not None:

                color_threshold = color_thresholds[0]
                color_index = 0
                while color_index < 9 and meaconer_biases[j][epoch] > color_threshold:
                    color_index += 1
                    color_threshold = color_thresholds[color_index]
                plt.scatter(x[0], y[0], linewidth=10, alpha=0.5, color=colors[color_index])

        for meaconer in range(nb_meaconers):
            x = []
            y = []
            for k in range(min(epoch + 1, nb_comet_meaconer)):
                if azimuths_meaconer[meaconer][epoch - k] is not None:
                    x.append(np.cos((np.pi/2 - azimuths_meaconer[meaconer][epoch - k])) *
                             ((90 - elevations_meaconer[meaconer][epoch - k] * RAD2DEG) / 90))
                    y.append(np.sin((np.pi/2 - azimuths_meaconer[meaconer][epoch - k])) *
                             ((90 - elevations_meaconer[meaconer][epoch - k] * RAD2DEG) / 90))
                    if len(x) > 1:
                        plt.plot([x[-2], x[-1]], [y[-2], y[-1]], "-r", linewidth=(1 - k/nb_comet_meaconer)*4)
            if len(x):
                plt.text(x[0], y[0], "M".format(meaconer), fontweight="semibold")

        plt.xlim([-1.3, 1.3])
        plt.axis("equal")
        plt.xticks([], [])
        plt.yticks([], [])
        mng = plt.get_current_fig_manager() # to uncomment to plot full screen
        mng.window.state('zoomed') # to uncomment to plot full screen
        plt.pause(0.000001)
    plt.show()


def display_skyplot_and_meaconer_positions_from_almanach_animation_with_color_code(user_positions, rx_epochs,
                                satellite_positions, meaconer_positions=[], meaconer_biases=[], label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    Depending on the noise parameters, the pseudodistances are distorted to match realistic behaviours.

    INPUTS :
        user_positions : the nb_epochs x 3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""

    # initialization of the Pseudodistances structure
    pseudodistances = Pseudodistances()
    pseudodistances.nb_epochs = len(rx_epochs)
    pseudodistances.label = label
    if L1:
        pseudodistances.frequencies["GPS"].append("L1")
    if L2:
        pseudodistances.frequencies["GPS"].append("L2")
    if L5:
        pseudodistances.frequencies["GPS"].append("L5")
    if E1:
        pseudodistances.frequencies["GAL"].append("E1")
    if E5:
        pseudodistances.frequencies["GAL"].append("E5")
    if R1:
        pseudodistances.frequencies["GLO"].append("R1")
    if R2:
        pseudodistances.frequencies["GLO"].append("R2")
    if B2:
        pseudodistances.frequencies["BDS"].append("B2")
    if B3:
        pseudodistances.frequencies["BDS"].append("B3")

    elevations = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]
    azimuths = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(72 * (GPS + GAL + GLO + BDS))]

    nb_meaconers = len(meaconer_positions)

    elevations_meaconer = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(nb_meaconers)]
    azimuths_meaconer = [[None for _ in range(pseudodistances.nb_epochs)] for _ in range(nb_meaconers)]

    colors = ['#069002', '#05BD00', '#5FDE01', '#B3EC10', '#EEF500', '#F5C800', '#F59C00', '#F56800', '#F53B00', '#F50000']
    color_thresholds = [0.8, 1.5, 3, 5, 8, 13, 21, 34, 55, 100]
    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance_from_almanach(user_positions[i], satellite_positions,
                                                               epoch, ELEVATION_MASK)
        pseudodistances.epochs.append({"time": epoch})

        for sat, sat_pos in euclidian_distances.items():
            pseudodistances.epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = pseudodistances.frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = pseudodistances.frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = pseudodistances.frequencies["GLO"]
            else:
                frequencies = pseudodistances.frequencies["BDS"]

            if len(frequencies):
                elevation, azimuth = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])
                pos = save_pos(sat, frequencies[0])
                elevations[pos][i] = elevation
                azimuths[pos][i] = azimuth

        for meaconer in range(nb_meaconers):
            if i < len(meaconer_positions[meaconer]):
                meaconer_position = meaconer_positions[meaconer][i]
            else:
                meaconer_position = meaconer_positions[meaconer][0]
            elevation_meaconer, azimuth_meaconer = conversion.ECEF2elevation_azimuth(user_positions[i],
                                                                                     meaconer_position)
            elevations_meaconer[meaconer][i] = elevation_meaconer
            azimuths_meaconer[meaconer][i] = azimuth_meaconer

    # plots the apparent position of the satellites in the sky
    reference_azimuths = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    reference_elevations = [-15, 0, 15, 30, 45, 60, 75]
    nb_points_reference = 360
    nb_comet = int(1 / (rx_epochs[1] - rx_epochs[0])) + 2
    nb_comet_meaconer = int(100 / (rx_epochs[1] - rx_epochs[0])) + 2
    plt.subplots(1, 1, figsize=(7, 5))

    for epoch in range(2, pseudodistances.nb_epochs):
        plt.clf()
        for el in reference_elevations:
            radius = ((90 - el) / 90) ** 1.05
            x = np.linspace(-radius, radius, nb_points_reference)
            y = np.sqrt(radius ** 2 - np.power(x, 2))
            plt.plot(x, y, "--k", alpha=0.5, linewidth=0.5)
            plt.plot(x, -y, "--k", alpha=0.5, linewidth=0.5)
            if el != reference_elevations[0]:
                plt.text(0.02, radius + 0.02, str(el) + "°")
        for az in reference_azimuths:
            theta = (90 - az) * DEG2RAD
            plt.plot([0, np.cos(theta) * 1.2], [0, np.sin(theta) * 1.2], "--k", alpha=0.5, linewidth=0.5)
            plt.text(np.cos(theta) * 1.24, np.sin(theta) * 1.24, str(az) + "°", rotation=theta * RAD2DEG - 90,
                     ha="center", va="center")

        for j in range(36 * (GPS + GAL + GLO + BDS)):
            x = []
            y = []
            sat = save_sat(j)
            if sat[0] == "G":
                col = "-b"
            elif sat[0] == "E":
                col = "-r"
            elif sat[0] == "R":
                col = "-g"
            else:
                col = "orange"
            for k in range(min(epoch + 1, nb_comet)):
                if azimuths[j][epoch - k] is not None:
                    x.append(np.cos((np.pi/2 - azimuths[j][epoch - k])) *
                             ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                    y.append(np.sin((np.pi/2 - azimuths[j][epoch - k])) *
                             ((90 - elevations[j][epoch - k] * RAD2DEG) / 90))
                    if len(x) > 1:
                        plt.plot([x[-2], x[-1]], [y[-2], y[-1]], col, linewidth=(1 - k/nb_comet)*3.5)
            if len(x):
                plt.text(x[0], y[0], sat[0:4], fontweight="semibold")

            if meaconer_biases[j][epoch] is not None:

                color_threshold = color_thresholds[0]
                color_index = 0
                while color_index < 9 and meaconer_biases[j][epoch] > color_threshold:
                    color_index += 1
                    color_threshold = color_thresholds[color_index]
                plt.scatter(x[0], y[0], linewidth=10, alpha=0.5, color=colors[color_index])

        for meaconer in range(nb_meaconers):
            x = []
            y = []
            for k in range(min(epoch + 1, nb_comet_meaconer)):
                if azimuths_meaconer[meaconer][epoch - k] is not None:
                    x.append(np.cos((np.pi/2 - azimuths_meaconer[meaconer][epoch - k])) *
                             ((90 - elevations_meaconer[meaconer][epoch - k] * RAD2DEG) / 90))
                    y.append(np.sin((np.pi/2 - azimuths_meaconer[meaconer][epoch - k])) *
                             ((90 - elevations_meaconer[meaconer][epoch - k] * RAD2DEG) / 90))
                    if len(x) > 1:
                        plt.plot([x[-2], x[-1]], [y[-2], y[-1]], "-r", linewidth=(1 - k/nb_comet_meaconer)*4)
            if len(x):
                plt.text(x[0], y[0], "M".format(meaconer), fontweight="semibold")

        plt.xlim([-1.3, 1.3])
        plt.axis("equal")
        plt.xticks([], [])
        plt.yticks([], [])
        mng = plt.get_current_fig_manager() # to uncomment to plot full screen
        mng.window.state('zoomed') # to uncomment to plot full screen
        plt.pause(0.000001)
    plt.show()


def generate_post_SBAS_correction_pseudoranges_from_almanach_with_meaconer_in_parallel(user_positions, rx_epochs,
                    satellite_positions, scenarios, grouped_scenarios, sampling_frequency=GENERATED_SAMPLING_FREQUENCY,
                                                                        bar=None, label="unnamed"):
    """this function generates the pseudodistances from all the visible satellites at epoch 'rx_epoch' of the receiver.
    The pseudoranges are distorted according to DO229E residual SBAS errors (ref : DO229E 2.5.10.3.1., December 2016)

    INPUTS :
        user_positions : the nb_epochsx3 ECEF user positions (in meters) at epochs 'rx_epochs'
        rx_epochs : the receiver epochs (in GPS time) at which to compute the received pseudodistances
        satellite_position : the structure containing the satellite positions from the sp3 file

    OUTPUTS :
        pseudodistances : the structure containing for all the visible satellites, the pseudodistances (in meters)
                          as received by the user at epochs 'rx_epochs'"""

    # initialization of the Pseudodistances structure
    for sc in range(len(grouped_scenarios)):
        scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"] = Pseudodistances()
        scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].nb_epochs = len(rx_epochs)
        scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].label = label
        if L1:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GPS"].append("L1")
        if L2:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GPS"].append("L2")
        if L5:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GPS"].append("L5")
        if E1:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GAL"].append("E1")
        if E5:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GAL"].append("E5")
        if R1:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GLO"].append("R1")
        if R2:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["GLO"].append("R2")
        if B2:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["BDS"].append("B2")
        if B3:
            scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].frequencies["BDS"].append("B3")

    # generation of the multipath code error
    multipath_code_errors = {}
    for sat, _ in satellite_positions.items():
        if sat[0] == "G":
            multipath_code_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GPS"]:
                multipath_code_errors[sat][freq] = rdm.normal(0, 1)
        if sat[0] == "E":
            multipath_code_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GAL"]:
                multipath_code_errors[sat][freq] = rdm.normal(0, 1)
        if sat[0] == "R":
            multipath_code_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GLO"]:
                multipath_code_errors[sat][freq] = rdm.normal(0, 1)
        if sat[0] == "C":
            multipath_code_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["BDS"]:
                multipath_code_errors[sat][freq] = rdm.normal(0, 1)

    # generation of the delta tropospheric zenith wet error
    tropo_errors = {}
    for sat, _ in satellite_positions.items():
        if sat != "time":
            tropo_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GPS"]:
                tropo_errors[sat][freq] = 0

    # generation of the UIRE ionospheric delay estimation
    uire_errors = {}
    for sat, _ in satellite_positions.items():
        if sat[0] == "G":
            uire_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GPS"]:
                uire_errors[sat][freq] = rdm.normal(0, 0.432)
        if sat[0] == "E":
            uire_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GAL"]:
                uire_errors[sat][freq] = rdm.normal(0, 0.432)
        if sat[0] == "R":
            uire_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GLO"]:
                uire_errors[sat][freq] = rdm.normal(0, 0.432)
        if sat[0] == "C":
            uire_errors[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["BDS"]:
                uire_errors[sat][freq] = rdm.normal(0, 0.432)

    # initialization of the delta_tau of the tracking loops
    previous_delta_tau = {}
    for sat, _ in satellite_positions.items():
        if sat != "time":
            previous_delta_tau[sat] = {}
            for freq in scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GPS"]:
                previous_delta_tau[sat][freq] = {}
                for sc in range(len(grouped_scenarios)):
                    previous_delta_tau[sat][freq][str(sc)] = None

    # initialization of the meaconer induced bias of the tracking loops
    meaconer_bias_save = [[[None for _ in range(scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].nb_epochs)]
                        for _ in range(36 * (GPS + GAL + GLO + BDS))] for _ in range(len(grouped_scenarios))]
    for sc in range(len(grouped_scenarios)):
        scenarios.spoofed[grouped_scenarios[sc]]["meaconer_biases"] = \
        [[None for _ in range(scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].nb_epochs)]
                        for _ in range(36 * (GPS + GAL + GLO + BDS))]

    # generation of the durations of availablilty of the satellites
    availability = {}
    visible_satellites = [0 for _ in range(72 * (GPS + GAL + GLO + BDS))]
    uire_errors_save = [[None for _ in range(scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].nb_epochs)]
                        for _ in range(72 * (GPS + GAL + GLO + BDS))]
    for sat, _ in satellite_positions.items():
        availability[sat] = 0

    for i, epoch in enumerate(rx_epochs):
        euclidian_distances = euclidian_distance_from_almanach(user_positions[i], satellite_positions,
                                                               epoch, ELEVATION_MASK)
        for sc in grouped_scenarios:
            scenarios.spoofed[sc]["pseudoranges"].epochs.append({"time": epoch})

        if i:
            delta_epoch = rx_epochs[i] - rx_epochs[i - 1]

        for sat, sat_pos in euclidian_distances.items():
            elevation = conversion.ECEF2elevation_azimuth(user_positions[i], sat_pos["pos"])[0]
            for sc in grouped_scenarios:
                scenarios.spoofed[sc]["pseudoranges"].epochs[i][sat] = {}
            if sat[0] == "G":
                frequencies = scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GPS"]
            elif sat[0] == "E":
                frequencies = scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GAL"]
            elif sat[0] == "R":
                frequencies = scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["GLO"]
            else:
                frequencies = scenarios.spoofed[grouped_scenarios[0]]["pseudoranges"].frequencies["BDS"]

            for freq in frequencies:

                if availability[sat]:

                    # generation of the new multipath code error:
                    multipath_code_errors[sat][freq] = 0.49 + 0.53 * np.exp(-elevation / (10 * DEG2RAD))

                    # generation of the new UIRE code error:
                    slang_factor = np.power(1 - np.power(EARTH_RADIUS * np.cos(elevation) / (EARTH_RADIUS + IONO_HEIGHT), 2), -0.5)

                    uire_code_std = np.sqrt((0.432 * slang_factor) ** 2 * (1 - np.exp(-2 *
                                    delta_epoch / 120)))
                    uire_errors[sat][freq] = np.exp(-delta_epoch /
                                    120) * uire_errors[sat][freq] + rdm.normal(0, uire_code_std)

                # tropospheric errors
                tropo_errors[sat][freq] = 0.12 * 1.001 / np.sqrt(0.002001 + np.sin(elevation) ** 2)

                # fast and long term corrections
                error_flt = 0.562

                # SBAS residual error
                sbas_error = rdm.normal(0, np.sqrt(error_flt ** 2 + uire_errors[sat][freq] ** 2
                                                   + tropo_errors[sat][freq] ** 2 + multipath_code_errors[sat][
                                                       freq] ** 2))

                # Doppler figure computation :
                if i:
                    user_velocity = (user_positions[i] - user_positions[i - 1]) * sampling_frequency
                else:
                    user_velocity = (user_positions[i + 1] - user_positions[i]) * sampling_frequency
                frequency = find_frequency_from_band(freq)
                doppler = frequency * np.dot(user_velocity - sat_pos["vel"],
                                             (sat_pos["pos"] - user_positions[i]) / sat_pos["distance"]) / C

                # C/N0 estimation
                cn0 = 113 - sat_pos["distance"] * 3.018e-6

                for sc in range(len(grouped_scenarios)):

                    # addition of the meaconer's induced bias
                    meaconer = scenarios.spoofed[grouped_scenarios[sc]]["meaconer"]

                    # # simulating the meaconer INSIDE the aircraft
                    # meaconer.ecef_position = [user_positions[i][0] + 0.1,
                    #                           user_positions[i][1] + 0,
                    #                           user_positions[i][2] + 0]

                    # computation of the GNSS signal delay of the spoofed signal (in seconds)
                    delta_tau = np.linalg.norm(sat_pos["pos"] - np.squeeze(np.transpose(meaconer.ecef_position))) / C -\
                                 sat_pos["distance"] / C + meaconer.delay * 1e-9 + \
                                 np.linalg.norm(user_positions[i] - np.squeeze(np.transpose(meaconer.ecef_position))) / C
                    frequency = find_frequency_from_band(freq)
                    lamda = C / frequency
                    delta_theta = np.mod(delta_tau, lamda) * 2 * np.pi / lamda
                    # computation of the free space loss (expressed in linear form)
                    elevation_meaconer = conversion.ECEF2elevation_azimuth(user_positions[i],
                                         np.squeeze(np.transpose(meaconer.ecef_position)))[0]
                    signal_attenuation = (lamda / 4 / np.pi / np.linalg.norm(user_positions[i] -
                                         np.squeeze(np.transpose(meaconer.ecef_position)))) ** 2 \
                                         * maximum_antenna_gain(elevation_meaconer)
                    autocorrelation_function = lambda tau: meaconer_bias_estimation.R_2(tau, BANDWIDTH)

                    meaconer_bias = meaconer_bias_estimation.compute_meaconer_bias(
                       meaconer_bias_estimation.discriminator_NEMLP, meaconer_bias_estimation.discriminator_atan2,
                       INTEGRATION_TIME, doppler + frequency, autocorrelation_function, INTER_CHIP_SPACING,
                       1, [delta_tau], [meaconer.delay * 1e-9], [signal_attenuation],
                       [10 ** (meaconer.gain / 10)], 1, CHIPPING_PERIOD, previous_delta_tau[sat][freq][str(sc)])

                    # old_meaconer_bias = meaconer_bias_estimation_old_version.compute_meaconer_bias(
                    #     delta_tau, meaconer.delay * 1e-9, signal_attenuation, 10 ** (meaconer.gain / 10),
                    #     previous_delta_tau[sat][freq][str(sc)])

                    sbas_error = 0

                    # if sat not in ['G11']:
                    #     meaconer_bias = [0, 0]

                    previous_delta_tau[sat][freq][str(sc)] = meaconer_bias[0]
                    pos = save_pos(sat, freq)
                    meaconer_bias_save[sc][pos][i] = meaconer_bias[0] * C
                    scenarios.spoofed[grouped_scenarios[sc]]["meaconer_biases"][pos][i] = meaconer_bias[0] * C

                    # generation of the new pseudodistances for the given satellite, frequency and epoch
                    scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].epochs[i][sat][freq] = \
                        {"code": sat_pos["distance"] + sbas_error + meaconer_bias[0] * C,
                            "phase": sat_pos["distance"] + sbas_error / 500 + meaconer_bias[1] * lamda,
                            "doppler": doppler, "CN0": cn0, "meaconer_code_bias": meaconer_bias[0] * C}
                    scenarios.spoofed[grouped_scenarios[sc]]["pseudoranges"].nb_pseudoranges += 1
                    bar()
                pos = save_pos(sat, freq)
                visible_satellites[pos] = 1
                uire_errors_save[pos][i] = uire_errors[sat][freq]




        visible_satellites_at_epoch = []
        for sat, _ in euclidian_distances.items():
            visible_satellites_at_epoch.append(sat)
        for sat, _ in satellite_positions.items():
            if sat in visible_satellites_at_epoch:
                availability[sat] += 1
            else:
                availability[sat] = 0


#########################
### TESTS AND DISPLAY ###
#########################

if __name__ == "__main__":
    satellite_positions = read_satellite_position(SP3_FILEPATH, GPS_ANTEX_FILEPATH, GAL_ANTEX_FILEPATH,
                                                  GLO_ANTEX_FILEPATH, BDS_ANTEX_FILEPATH)

    track = waypoints.load_track(TRACK)
    jump = int(track.freq / GENERATED_SAMPLING_FREQUENCY)
    print("Please wait, GNSS generation takes about {} min\n".format(
        int(0.001 * (GPS + GAL + GLO + BDS) * track.nb_points / jump)))
    if not GENERATED_SAMPLING_FREQUENCY:
        times = track.time
        user_positions = track.pos
    else:
        times = track.time[0:-1:jump]
        user_positions = track.pos[0:-1:jump]
    pseudodistances = generate_post_SBAS_correction_pseudodistances(user_positions, times, satellite_positions)
    print(pseudodistances)
    pseudodistances.save_in_msr(SAVE_TRACK)
