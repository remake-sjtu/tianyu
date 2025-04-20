import pandas as pd
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.wcs.utils import wcs_to_celestial_frame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def fibonacci_covering(theta_rad):
    n_points = int(np.ceil(4 / (theta_rad ** 2)))
    golden_angle = np.pi * (3 - np.sqrt(5))
    centers = []
    for i in range(n_points):
        z = 1 - 2 * i / float(n_points - 1)
        radius = np.sqrt(1 - z * z)
        theta = golden_angle * i
        centers.append((np.cos(theta) * radius, np.sin(theta) * radius, z))
    return np.array(centers)


def generate_sky_WCS(NAXIS1, NAXIS2, pixel_scale, theta, xflip=False):
    angular_radius_deg = pixel_scale / 2 * np.min([NAXIS1, NAXIS2])  # Convert to degrees
    theta_rad = np.radians(angular_radius_deg)
    centers = np.array(fibonacci_covering(theta_rad))
    print(f"Generated {len(centers)} cap centers.")

    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]

    ra = np.arctan2(y, x) * 180 / np.pi + 180
    dec = np.arcsin(z) * 180 / np.pi
    radec = np.hstack((ra.reshape(-1, 1), dec.reshape(-1, 1)))
    # Have some redundant info if it is for the same instrumentation
    WCS_info_dict = {"sky_index": [], "NAXIS": [], "NAXIS1": [], "NAXIS2": [], "CTYPE1": [], "CTYPE2": [], "CRVAL1": [],
                     "CRVAL2": [], "CRPIX1": [], "CRPIX2": [], "CD1_1": [], "CD1_2": [], "CD2_1": [], "CD2_2": []}
    i = 1
    for ra_this, dec_this in radec:
        if xflip:
            factor = -1
        else:
            factor = 1
        CD_mat = np.array([[pixel_scale * factor, 0], [0, pixel_scale]])
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
        WCS_info_dict["CRPIX1"].append(NAXIS1 / 2)
        WCS_info_dict["CRPIX2"].append(NAXIS2 / 2)
        WCS_info_dict["CD1_1"].append(CD_mat[0, 0])
        WCS_info_dict["CD1_2"].append(CD_mat[0, 1])
        WCS_info_dict["CD2_1"].append(CD_mat[1, 0])
        WCS_info_dict["CD2_2"].append(CD_mat[1, 1])
        if i == 10:
            break
    WCS_info_dict = pd.DataFrame(WCS_info_dict)
    return WCS_info_dict



def calculate_region_visibility(wcs, image_shape, obs_time, obs_location, stars_icrs, mags, mag_limit=20.0):
    try:
        center = wcs.pixel_to_world(image_shape[0] // 2, image_shape[1] // 2)
        altaz = center.transform_to(AltAz(location=obs_location, obstime=obs_time))
        if altaz.alt < 0 * u.deg:
            return (False, 0, 0.0)
    except:
        return (False, 0, 0.0)

    try:
        frame = wcs_to_celestial_frame(wcs)
        stars = stars_icrs.transform_to(frame)
        x, y = wcs.world_to_pixel(stars)
    except:
        return (False, 0, 0.0)

    valid = (x >= 0) & (x < image_shape[0]) & (y >= 0) & (y < image_shape[1]) & (mags <= mag_limit)
    weights = 10 ** (-0.4 * mags[valid])
    return (True, np.sum(valid), np.sum(weights))


if __name__ == "__main__":

    # 生成WCS参数并保存
    NAXIS1, NAXIS2 = 9576, 6388
    pixel_scale = 3.76 / 2563000 * 180 / np.pi  # 转换为度/像素
    wcs_df = generate_sky_WCS(NAXIS1, NAXIS2, pixel_scale, theta=0)
    wcs_df.to_csv('sky_WCS.csv', index=False)

    # 模拟星表数据
    np.random.seed(42)
    N_stars = 1000000
    ra = np.random.uniform(0, 360, N_stars)
    dec = np.random.uniform(-90, 90, N_stars)
    mags = np.random.uniform(1, 20, N_stars)
    stars_icrs = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

    # 设置观测参数
    obs_location = EarthLocation(lat=33.3563 * u.deg, lon=-116.8650 * u.deg, height=1712 * u.m)
    obs_time = Time('2024-01-01 08:00:00')

    # 处理每个WCS区域
    results = []
    for _, row in pd.read_csv('sky_WCS.csv').iterrows():
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [row['CRPIX1'], row['CRPIX2']]
        wcs.wcs.cd = [[row['CD1_1'], row['CD1_2']], [row['CD2_1'], row['CD2_2']]]
        wcs.wcs.crval = [row['CRVAL1'], row['CRVAL2']]
        wcs.wcs.ctype = [row['CTYPE1'], row['CTYPE2']]

        visible, count, weight = calculate_region_visibility(
            wcs=wcs,
            image_shape=(row['NAXIS1'], row['NAXIS2']),
            obs_time=obs_time,
            obs_location=obs_location,
            stars_icrs=stars_icrs,
            mags=mags
        )

        results.append({
            'sky_index': row['sky_index'],
            'visible': visible,
            'star_count': count,
            'weight_sum': weight
        })

    # 输出结果
    result_df = pd.DataFrame(results)
    print(result_df)
