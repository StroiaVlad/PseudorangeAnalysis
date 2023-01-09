######################################
###    AUTHOR : MATHIEU HUSSONG    ###
###   ENAC TELECOM LAB COPYRIGHT   ###
###        DATE : 07/12/2021       ###
###          VERSION : 2           ###
######################################

# THIS CODE PROVIDES ALL THE UNIT CONVERSIONS THAT ARE USEFUL TO MASTER GNSS POSITIONING, IMU MECHANIZATION
# AND HYBRIDIZATIONS


#######################
### LIBRARY IMPORTS ###
#######################

import numpy as np


#################
### CONSTANTS ###
#################

DEG2RAD = np.pi / 180  # to convert from degrees to radians
RAD2DEG = 180 / np.pi  # to convert from radians to degrees
WGS84_A = 6378137  # the semi-major axis of the WGS84 model
WGS84_B = 6356752.31424518  # the semi-minor axis of the WGS84 model
WGS84_F = 1 / 298.257223563  # the flatness of the WGS84 model, we also have WGS84_F = (WGS84_A - WGS84_B) / WGS84_A
WGS84_E = np.sqrt((WGS84_A**2 - WGS84_B**2) / WGS84_A**2)  # the first eccentricity of the WGS84 model
WGS84_E_PRIME = np.sqrt((WGS84_A**2 - WGS84_B**2) / WGS84_B**2)  # the second eccentricity of the WGS84 model


#################
### FUNCTIONS ###
#################

def gregorian2GPStime(year, month, day, hour, min, sec, leap_seconds=18):
    """this function returns the GPS time (= the number of seconds passed from 01/01/0980 at 00:00:00)
    based on a gregorian date
    OUTPUTS : nos = GPS time (number of seconds)
              tod = number of seconds past last midnight (time of day)
    WARNING : don't try to understand this function, it's based on a C translation that is not intuitive.
    The function works fine, trust me."""

    hour_decimel = hour + min / 60 + sec / 3600
    if month <= 2:
        y = year - 1
        m = month + 12
    else:
        y = year
        m = month
    j = np.floor(365.25 * y)
    j = j + np.floor(30.6001 * (m + 1))
    j = j + day
    j = j + hour_decimel / 24
    j = j + 1720981.5
    num_seconds_day = 86400
    num_seconds_week = 86400 * 7
    a = np.floor(j + 0.5)
    b = a + 1537
    c = np.floor((b - 122.1) / 365.25)
    d = np.floor(365.25 * c)
    e = np.floor((b - d) / 30.6001)
    D = b
    D = D - d
    D = D - np.floor(30.6001 * e)
    D = D + (j + 0.5) % 1
    v = np.floor(j + 0.5)
    N = np.mod(v, 7)
    GPS_week = np.floor((j - 2444244.5) / 7)
    sow = (N + 1 + D%1) * num_seconds_day
    sow = np.mod(sow, num_seconds_week)
    nos = (GPS_week * num_seconds_week) + sow + leap_seconds
    tod = np.mod(nos, num_seconds_day)
    return nos, tod


def ECEF2LLA(pos_ECEF):
    """This function converts the ECEF position into the LLA one.
    %   INPUTS :
    %       pos_ECEF :  a 3x1 or 1x3 vector with the ECEF coordinates (in meter) of the position to convert
    %   OUTPUTS : a 1x3 vector [phi, lamda, h] with the LLA position
    %       phi : the latitude (in rad) of the position in LLA
    %       lamda : the longitude (in rad) of the position in LLA
    %       h : the altitude above the sea level (in meter) of the position"""
    p = np.sqrt(pos_ECEF[0]**2 + pos_ECEF[1]**2)
    if not p:
        lamda = 0
        phi = np.sign(pos_ECEF[2]) * np.pi / 2
        h = np.abs(pos_ECEF[2]) - WGS84_B
    else:
        theta = np.arctan(pos_ECEF[2] * WGS84_A / p / WGS84_B)
        if not pos_ECEF[0]:
            lamda = np.sign(pos_ECEF[1]) * np.pi / 2
        else:
            lamda = np.arctan(pos_ECEF[1] / pos_ECEF[0])
        phi = np.arctan((pos_ECEF[2] + WGS84_E_PRIME**2 * WGS84_B * np.sin(theta)**3) /
                        (p - WGS84_E**2 * WGS84_A * np.cos(theta)**3))
        n = WGS84_A / np.sqrt(1 - WGS84_E ** 2 * np.sin(phi) ** 2)
        if not np.cos(phi):
            h = np.abs(pos_ECEF[2]) - WGS84_B
        else:
            h = p / np.cos(phi) - n
    return np.array([phi, lamda, h])


def LLA2ECEF(pos_LLA):
    """This function converts the LLA position into the ECEF one.
    %   INPUTS :
    %       pos_LLA :  a 1x3 vector [phi, lamda, h] with the LLA position of the position to convert
    %           phi : the latitude (in rad) of the position in LLA
    %           lamda : the longitude (in rad) of the position in LLA
    %           h : the altitude above the sea level (in meter) of the position
    %   OUTPUTS : a 1x3 vector with the ECEF coordinates (in meter) of the position"""
    n = WGS84_A / np.sqrt(1 - WGS84_E ** 2 * np.sin(pos_LLA[0])**2)
    x = (n + pos_LLA[2]) * np.cos(pos_LLA[0]) * np.cos(pos_LLA[1])
    y = (n + pos_LLA[2]) * np.cos(pos_LLA[0]) * np.sin(pos_LLA[1])
    z = (WGS84_B**2 / WGS84_A**2 * n + pos_LLA[2]) * np.sin(pos_LLA[0])
    return np.array([x, y, z])


def ECEF2body(ecef_position, heading, attitude, roll):
    """returns the 3x3 ECEF to body rotation matrix that transforms any 3x1 ECEF vector into a 3x1 body vector.
    The body frame is defined with its first vector parallel to the plane fuselage, from the tail to the nose ;
    its second vector parallel to the right wing, from the left wing tip to the right wing tip ;
    its third vector completing the direct orthogonal frame.
    %   INPUTS :
    %       ecef_position : the ECEF-coordinates of the user (in meters)
    %       heading : the heading of the body (in radians, expressed clockwise from the true north)
    %       attitude : the attitude (assiette) angle of the body (in radians, positive when nose up)
    %       roll : the roll (banking) angle of the body (in radians, positive when right turn)

    %   OUTPUTS :
    %       rotation_matrix : the 3x3 rotation matrix that transforms ECEF coordinates into body coordinated as
                              body_vector = rotation_matrix * ecef_vector"""
    [phi, lamda, _] = ECEF2LLA(ecef_position)
    rotation_ecef2enu = np.array([[-np.sin(lamda), np.cos(lamda), 0],
                                    [-np.sin(phi)*np.cos(lamda),
                             -np.sin(phi)*np.sin(lamda), np.cos(phi)],
                            [np.cos(phi) * np.cos(lamda),
                             np.cos(phi) * np.sin(lamda), np.sin(phi)]])

    rotation_heading = np.array(
        [[np.sin(heading), np.cos(heading), 0], [np.cos(heading), -np.sin(heading), 0], [0, 0, -1]])
    rotation_attitude = np.array(
        [[np.cos(attitude), 0, -np.sin(attitude)], [0, 1, 0], [np.sin(attitude), 0, np.cos(attitude)]])
    rotation_roll = np.array(
        [[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
    rotation_matrix = np.dot(rotation_roll, np.dot(rotation_attitude,
                            np.dot(rotation_heading, rotation_ecef2enu)))
    return rotation_matrix


def ECEF2ENU(beacon_position, phi_reference, lamda_reference):
    """This function converts the ECEF position into an ENU position. The
    conversion is done with respect to a reference point (needed for the ENU
    conversion).

    %   INPUTS :
    %       beacon_position : the 3x1 ECEF position to convert into the ENU frame (in meters)
    %       phi_reference : the latitude (in radians) of the reference point
    %       lambda_reference : the longitude (in radians) of the reference point

    %   OUTPUT :
    %       enu : the 3x1 position in the ENU frame (whose origin is the center of
    %           the Earth and whose axes point towards the reference point, the
    %           east of the reference point, and the North of the reference
    %           point) in meters"""
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, :] = [-np.sin(lamda_reference), np.cos(lamda_reference), 0]
    rotation_matrix[1, :] = [-np.sin(phi_reference)*np.cos(lamda_reference),
                             -np.sin(phi_reference)*np.sin(lamda_reference), np.cos(phi_reference)]
    rotation_matrix[2, :] = [np.cos(phi_reference) * np.cos(lamda_reference),
                             np.cos(phi_reference) * np.sin(lamda_reference), np.sin(phi_reference)]

    return np.dot(rotation_matrix, beacon_position)


def ECI2ECEF(eci_position, time_univ):
    """This function converts coordinates from the ECI frame into the ECEF frame.
    A python built-in function exists and can be used instead if the
    adequate toolbox is usable.
    This function do not take into account the nutation and precession effects.

    %   INPUTS :
    %       eci_position : position in ECI to convert in ECEF (in meters)
    %       time_univ : universal time (in seconds)

    %   OUTPUT :
    %       ecef_position : position in the ECEF-frame (in meters)"""
    theta = 7.2921151467e-5 * time_univ
    ecef2eci_rotmat = [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    ecef_position = np.dot(np.linalg.inv(ecef2eci_rotmat), eci_position)
    return ecef_position


def ECEF02ECEF(ecef0_position, time):
    """This function converts coordinates from the initial ECEF frame into the ECEF frame.

    %   INPUTS :
    %       eci_position : position in ECI to convert in ECEF (in meters)
    %       time_univ : universal time (in seconds)

    %   OUTPUT :
    %       ecef_position : position in the ECEF-frame (in meters)"""
    theta = 7.2921151467e-5 * time
    ecef2eci_rotmat = [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    ecef_position = np.dot(np.linalg.inv(ecef2eci_rotmat), ecef0_position)
    return ecef_position


def ECEF2ECEF0(ecef0_position, time):
    """This function converts coordinates from the ECEF frame to the initial ECEF frame.

    %   INPUTS :
    %       eci_position : position in ECI to convert in ECEF (in meters)
    %       time_univ : universal time (in seconds)

    %   OUTPUT :
    %       ecef_position : position in the ECEF-frame (in meters)"""
    theta = 7.2921151467e-5 * time
    ecef2eci_rotmat = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    ecef_position = np.dot(np.linalg.inv(ecef2eci_rotmat), ecef0_position)
    return ecef_position


def CoM2ECEF(sun_position_ecef, satellite_position_ecef):
    """This function computes the rotation matrix from the CoM satellite frame to the ECEF frame

    %   INPUTS :
    %       Sun_position_ecef : the Sun position in the ECEF frame
    %       satellite_position_ecef : the satellite position in the ECEF frame

    %   OUTPUT :
    %       com2ecef : the rotation matrix between the CoM frame to the ECEF frame"""
    distance_earth_satellite = np.linalg.norm(satellite_position_ecef)
    sat_pos_unit_vector = [-satellite_position_ecef[0] / distance_earth_satellite,
              -satellite_position_ecef[1] / distance_earth_satellite,
              -satellite_position_ecef[2] / distance_earth_satellite]
    sun_sat_vector = sun_position_ecef - satellite_position_ecef
    e_sy_ki = np.array(np.cross(sat_pos_unit_vector, sun_sat_vector))
    norm_e_sy_ki = np.linalg.norm(e_sy_ki)
    e_sy_k = np.array(e_sy_ki / norm_e_sy_ki)
    e_sx_k = np.array(np.cross(e_sy_k, sat_pos_unit_vector))
    com2ecef = np.transpose([e_sx_k, e_sy_k, sat_pos_unit_vector])
    return np.array(com2ecef)


def ECEF2elevation_azimuth(user_position, beacon_position):
    """This function computes the azimuth and the elevation of a point given by
    its ECEF coordinates (beacon_position) with respect to a reference location given
    by user_position. The function returns the elevation, and the azimuth of
    the beacon location as seen by the user

    %   INPUTS :
    %       user_position : the ECEF-coordinates of the reference user (in meters)
    %       beacon_position : the ECEF_coordinates of the beacon / the satellite (in meters)

    %   OUTPUTS :
    %       elevation : elevation (rad) of the satellite with respect to the receiver
    %       azimuth : azimuth (rad) of the satellite with respect to the receiver
                        azimuth is the oriented trigonometric angle (not with respect to the true north)"""
    phi, lamda, alt = ECEF2LLA(user_position)
    los_vector = beacon_position - np.transpose(user_position)
    enu_position = ECEF2ENU(los_vector, phi, lamda)
    los_range = np.linalg.norm(los_vector)
    azimuth = np.arctan2(enu_position[0], enu_position[1])
    elevation = np.arccos(np.sqrt(enu_position[0] ** 2 + enu_position[1] ** 2) / los_range) * np.sign(enu_position[2])
    return elevation, azimuth


# def body2nav(euler_angles):
#     psi = euler_angles[0]
#     theta = euler_angles[1]
#     phi = euler_angles[2]
#     return np.array([[np.cos(psi)*np.cos(phi) - np.sin(psi)*np.cos(theta)*np.sin(phi),
#                       -np.cos(psi)*np.sin(phi) - np.sin(psi)*np.cos(theta)*np.cos(phi),
#                       np.sin(psi)*np.sin(theta)],
#                      [np.sin(psi) * np.cos(phi) + np.cos(psi) * np.cos(theta) * np.sin(phi),
#                       -np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(theta) * np.cos(phi),
#                       -np.cos(psi) * np.sin(theta)],
#                      [np.sin(theta)*np.sin(phi), np.sin(theta)*np.cos(phi), np.cos(theta)]])
#
#
# def nav2body(euler_angles):
#     psi = euler_angles[0]
#     theta = euler_angles[1]
#     phi = euler_angles[2]
#     return np.transpose(np.array([[np.cos(psi) * np.cos(phi) - np.sin(psi) * np.cos(theta) * np.sin(phi),
#                       -np.cos(psi) * np.sin(phi) - np.sin(psi) * np.cos(theta) * np.cos(phi),
#                       np.sin(psi) * np.sin(theta)],
#                      [np.sin(psi) * np.cos(phi) + np.cos(psi) * np.cos(theta) * np.sin(phi),
#                       -np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(theta) * np.cos(phi),
#                       -np.cos(psi) * np.sin(theta)],
#                      [np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta)]]))
#
#
# def partial_nav2body_nominal(euler_angles):
#     """computes the partial derivative of the rotation matrix from navigation to nominal body frame,
#     evaluated at the nominal body position"""
#     psi = euler_angles[0]
#     theta = euler_angles[1]
#     phi = euler_angles[2]
#     return np.array([[-np.sin(psi) * np.cos(phi) - np.cos(psi) * np.cos(theta) * np.sin(phi),
#                       np.cos(psi) * np.cos(phi) - np.sin(psi) * np.cos(theta) * np.sin(phi),
#                       0],
#                      [np.sin(psi) * np.sin(phi) - np.cos(psi) * np.cos(theta) * np.cos(phi),
#                       -np.cos(psi) * np.sin(phi) - np.sin(psi) * np.cos(theta) * np.cos(phi),
#                       0],
#                      [np.cos(psi) * np.sin(theta),
#                       np.sin(psi) * np.sin(theta),
#                       0],
#                      [np.sin(psi) * np.sin(theta) * np.sin(phi),
#                       -np.cos(psi) * np.sin(theta) * np.sin(phi),
#                       np.cos(theta) * np.sin(phi)],
#                      [np.sin(psi) * np.sin(theta) * np.cos(phi),
#                       -np.cos(psi) * np.sin(theta) * np.cos(phi),
#                       np.cos(theta) * np.cos(phi)],
#                      [np.sin(psi) * np.cos(theta),
#                       -np.cos(psi) * np.cos(theta),
#                       -np.sin(theta)],
#                      [-np.cos(psi) * np.sin(phi) - np.sin(psi) * np.cos(theta) * np.cos(phi),
#                       -np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(theta) * np.cos(phi),
#                       np.sin(theta) * np.cos(phi)],
#                      [-np.cos(psi) * np.cos(phi) + np.sin(psi) * np.cos(theta) * np.sin(phi),
#                       -np.sin(psi) * np.cos(phi) - np.cos(psi) * np.cos(theta) * np.sin(phi),
#                       -np.sin(theta) * np.sin(phi)],
#                      [0,
#                       0,
#                       0]])
#
#
# def partial_body2nav_nominal(euler_angles):
#     """computes the partial derivative of the rotation matrix from body to navigation frame,
#     evaluated at the nominal body position"""
#     psi = euler_angles[0]
#     theta = euler_angles[1]
#     phi = euler_angles[2]
#     return np.transpose(np.array([[-np.sin(psi) * np.cos(phi) - np.cos(psi) * np.cos(theta) * np.sin(phi),
#                       np.cos(psi) * np.cos(phi) - np.sin(psi) * np.cos(theta) * np.sin(phi),
#                       0],
#                      [np.sin(psi) * np.sin(phi) - np.cos(psi) * np.cos(theta) * np.cos(phi),
#                       -np.cos(psi) * np.sin(phi) - np.sin(psi) * np.cos(theta) * np.cos(phi),
#                       0],
#                      [np.cos(psi) * np.sin(theta),
#                       np.sin(psi) * np.sin(theta),
#                       0],
#                      [np.sin(psi) * np.sin(theta) * np.sin(phi),
#                       -np.cos(psi) * np.sin(theta) * np.sin(phi),
#                       np.cos(theta) * np.sin(phi)],
#                      [np.sin(psi) * np.sin(theta) * np.cos(phi),
#                       -np.cos(psi) * np.sin(theta) * np.cos(phi),
#                       np.cos(theta) * np.cos(phi)],
#                      [np.sin(psi) * np.cos(theta),
#                       -np.cos(psi) * np.cos(theta),
#                       -np.sin(theta)],
#                      [-np.cos(psi) * np.sin(phi) - np.sin(psi) * np.cos(theta) * np.cos(phi),
#                       -np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(theta) * np.cos(phi),
#                       np.sin(theta) * np.cos(phi)],
#                      [-np.cos(psi) * np.cos(phi) + np.sin(psi) * np.cos(theta) * np.sin(phi),
#                       -np.sin(psi) * np.cos(phi) - np.cos(psi) * np.cos(theta) * np.sin(phi),
#                       -np.sin(theta) * np.sin(phi)],
#                      [0,
#                       0,
#                       0]]))
#
#
# def partial_inv_Q_BE(ecef_position, time, euler_angles):
#     return np.transpose(np.dot(ECEF2ECEF0(ecef_position, time), partial_body2nav_nominal(euler_angles)))
