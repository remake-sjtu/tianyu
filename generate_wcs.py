import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
def fibonacci_covering(theta_rad):
    assert 0 < theta_rad < np.pi / 2, "Angular radius should be in (0, π/2) radians."
    n_points = int(np.ceil(4 / (theta_rad ** 2)))

    centers = []
    golden_angle = np.pi * (3 - np.sqrt(5))

    for i in range(n_points):
        z = 1 - 2 * i / float(n_points - 1)
        radius = np.sqrt(1 - z * z)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        centers.append((x, y, z))

    return centers

def plot_sphere_with_caps(centers):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])

    # Plot unit sphere
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.2, linewidth=0)

    # Plot cap centers
    xs, ys, zs = zip(*centers)
    ax.scatter(xs, ys, zs, color='red', s=1, label='Cap Centers')

    ax.set_title("Spherical Cap Centers on Unit Sphere")
    ax.legend()
    plt.tight_layout()
    plt.show()

# Example usage
def generate_sky_WCS(NAXIS1,NAXIS2,pixel_scale,theta ,xflip=False):
   
    angular_radius_deg = pixel_scale/2*np.min([NAXIS1,NAXIS2])  # Convert to degrees
    theta_rad = np.radians(angular_radius_deg)
    centers = np.array(fibonacci_covering(theta_rad))
    print(f"Generated {len(centers)} cap centers.")
    
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]

    ra = np.arctan2(y, x) * 180 / np.pi+180
    dec = np.arcsin(z) * 180 / np.pi
    radec = np.hstack((ra.reshape(-1,1), dec.reshape(-1,1)))
    # Have some redundant info if it is for the same instrumentation
    WCS_info_dict = {"sky_index":[],"NAXIS":[],"NAXIS1":[],"NAXIS2":[],"CTYPE1":[],"CTYPE2":[],"CRVAL1":[],"CRVAL2":[],"CRPIX1":[],"CRPIX2":[],"CD1_1":[],"CD1_2":[],"CD2_1":[],"CD2_2":[]}
    i = 1
    for ra_this,dec_this in radec:
        if xflip:
            factor = -1
        else:
            factor = 1
        CD_mat = np.array([[pixel_scale*factor, 0], [0, pixel_scale]])
        rotatio_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        CD_mat = np.dot(rotatio_matrix, CD_mat)
        # The WCS info
        WCS_info_dict["sky_index"].append(i)
        i += 1
        WCS_info_dict["NAXIS"].append(2)
        WCS_info_dict["NAXIS1"].append(NAXIS1)
        WCS_info_dict["NAXIS2"].append(NAXIS2)
        WCS_info_dict["CTYPE1"].append("RA---TAN")
        WCS_info_dict["CTYPE2"].append("DEC--TAN")
        WCS_info_dict["CRVAL1"].append(ra_this)
        WCS_info_dict["CRVAL2"].append(dec_this)
        WCS_info_dict["CRPIX1"].append(NAXIS1/2)
        WCS_info_dict["CRPIX2"].append(NAXIS2/2)
        WCS_info_dict["CD1_1"].append(CD_mat[0,0])
        WCS_info_dict["CD1_2"].append(CD_mat[0,1])
        WCS_info_dict["CD2_1"].append(CD_mat[1,0])
        WCS_info_dict["CD2_2"].append(CD_mat[1,1])
    WCS_info_dict = pd.DataFrame(WCS_info_dict)
    return WCS_info_dict
        



NAXIS1 = 9576 # Number of pixel of our telescope
NAXIS2 = 6388 # Number of pixel of our telescope
pixel_scale = 3.76 / 2563000 * 180 / np.pi # deg per pixel
theta = 0
pd_WCS = generate_sky_WCS(NAXIS1,NAXIS2,pixel_scale,theta)
pd_WCS.to_csv('sky_WCS.csv', index=False)