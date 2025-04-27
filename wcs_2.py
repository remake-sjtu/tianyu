import pandas as pd
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body
from astropy.time import Time
import astropy.units as u
from astropy.wcs.utils import wcs_to_celestial_frame, proj_plane_pixel_scales
from astroquery.gaia import Gaia


def query_gaia_stars(center_ra, center_dec, search_radius, mag_limit=14.0):
    """优化后的Gaia星表查询"""
    query = f"""
    SELECT TOP 100000 
      ra, dec, phot_g_mean_mag, pmra, pmdec,
      phot_bp_mean_mag, phot_rp_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
      POINT('ICRS', ra, dec),
      CIRCLE('ICRS', {center_ra}, {center_dec}, {search_radius})
    )
    AND phot_g_mean_mag < {mag_limit}
    AND astrometric_params_solved = 31
    """
    return Gaia.launch_job_async(query).get_results()


def calculate_visibility(wcs, image_shape, obs_time, location, stars_icrs, mags, mag_limit):
    try:
        # 检查天区可见性
        frame = wcs_to_celestial_frame(wcs)
        stars = stars_icrs.transform_to(frame)
        x, y = wcs.world_to_pixel(stars)

        # 计算有效区域
        valid = (x >= 0) & (x < image_shape[0]) & \
                (y >= 0) & (y < image_shape[1]) & \
                (mags <= mag_limit)

        # 计算权重
        weights = 10 ** (-0.4 * mags[valid])
        return (True, np.sum(valid), np.sum(weights))
    except Exception as e:
        return (False, 0, 0.0)


def is_in_target_regions(altaz_coord, target_regions):
    """检查坐标是否在目标区域内"""
    az = altaz_coord.az.deg
    alt = altaz_coord.alt.deg
    for (az_min, az_max, alt_min, alt_max) in target_regions:
        if (alt_min <= alt <= alt_max) and (
                (az_min <= az <= az_max) if az_min <= az_max
                else (az >= az_min or az <= az_max)):
            return True
    return False


def get_target_regions(obs_time, obs_lon, obs_lat):
    location = EarthLocation(lon=obs_lon * u.deg, lat=obs_lat * u.deg)
    time = Time(obs_time)

    sun_altaz = get_sun(time).transform_to(AltAz(location=location, obstime=time))
    anti_sun = {
        'az': (sun_altaz.az + 180 * u.deg) % 360 * u.deg,
        'alt_range': (70 * u.deg, 90 * u.deg),
        'az_width': 10 * u.deg
    }

    moon_altaz = get_body("moon", time).transform_to(AltAz(location=location, obstime=time))
    anti_moon = {
        'az': (moon_altaz.az + 180 * u.deg) % 360 * u.deg,
        'alt_range': (60 * u.deg, 90 * u.deg),
        'az_width': 10 * u.deg
    }

    return [
        (anti_sun['az'] - anti_sun['az_width'] / 2,
         anti_sun['az'] + anti_sun['az_width'] / 2,
         *anti_sun['alt_range']),
        (anti_moon['az'] - anti_moon['az_width'] / 2,
         anti_moon['az'] + anti_moon['az_width'] / 2,
         *anti_moon['alt_range'])
    ]


def is_time_suitable(obs_time, location):
    sun_alt = get_sun(obs_time).transform_to(
        AltAz(location=location, obstime=obs_time)).alt
    return -10 * u.deg <= sun_alt <= -6 * u.deg


def observe_system(obs_time, obs_lon, obs_lat, wcs_df, mag_limit=20.0, buffer=0.2):
    """
    观测系统主函数
    :param obs_time: 观测时间（可解析的时间字符串或Time对象）
    :param obs_lon: 观测点经度（十进制度）
    :param obs_lat: 观测点纬度（十进制度）
    :param wcs_df: 包含WCS参数的DataFrame，需包含以下列 具体可以看本仓库中的csv文件
                   sky_index, NAXIS1, NAXIS2, CTYPE1, CTYPE2,
                   CRVAL1, CRVAL2, CRPIX1, CRPIX2, CD1_1, CD1_2, CD2_1, CD2_2
    :param mag_limit: 最大星等限制
    :param buffer: Gaia查询缓冲系数（0-1之间）
    :return: 包含观测结果的DataFrame
    """
    suitable_ra = 0
    suitable_dec = 0
    min_score = 10000
    # 初始化天文参数
    obs_time = Time(obs_time)
    location = EarthLocation(lon=obs_lon * u.deg, lat=obs_lat * u.deg)

    # 检查观测时间有效性
    if not is_time_suitable(obs_time, location):
        return pd.DataFrame(columns=['sky_index', 'visible', 'star_count', 'weight_sum', 'ra', 'dec', 'az', 'alt'])

    # 获取目标观测区域 反太阳区 反月亮区
    target_regions = get_target_regions(obs_time, obs_lon, obs_lat)
    target_regions_deg = [(r[0].deg, r[1].deg, r[2].deg, r[3].deg) for r in target_regions]

    results = []

    # 处理每个WCS区域
    for _, row in wcs_df.iterrows():
        # 解析WCS参数
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = [row['CTYPE1'], row['CTYPE2']]
        wcs.wcs.crval = [row['CRVAL1'], row['CRVAL2']]
        wcs.wcs.crpix = [row['CRPIX1'], row['CRPIX2']]
        wcs.wcs.cd = [[row['CD1_1'], row['CD1_2']],
                      [row['CD2_1'], row['CD2_2']]]

        # 计算视场参数
        try:
            pixel_scales = proj_plane_pixel_scales(wcs) * u.deg
            fov_width = row['NAXIS1'] * pixel_scales[0]
            fov_height = row['NAXIS2'] * pixel_scales[1]
            radius = np.sqrt((fov_width / 2) ** 2 + (fov_height / 2) ** 2) * (1 + buffer)

            # 获取准确中心坐标
            center = wcs.pixel_to_world(row['CRPIX1'], row['CRPIX2'])
            center_altaz = center.transform_to(AltAz(location=location, obstime=obs_time))
        except Exception as e:
            print(f"WCS解析失败: {e}")
            continue

        # 检查是否在目标区域内
        if not is_in_target_regions(center_altaz, target_regions_deg):
            continue

        # 查询Gaia星表
        try:
            stars_table = query_gaia_stars(
                center_ra=center.ra.deg,
                center_dec=center.dec.deg,
                search_radius=radius.value,
                mag_limit=mag_limit
            )
            stars_icrs = SkyCoord(ra=stars_table['ra'] * u.deg,
                                  dec=stars_table['dec'] * u.deg)
            mags = stars_table['phot_g_mean_mag'].data
        except Exception as e:
            print(f"星表查询失败: {e}")
            continue

        # 计算可见性指标
        visible, count, weight = calculate_visibility(
            wcs=wcs,
            image_shape=(row['NAXIS1'], row['NAXIS2']),
            obs_time=obs_time,
            location=location,
            stars_icrs=stars_icrs,
            mags=mags,
            mag_limit=mag_limit
        )
        # 记录最佳的结果
        # 记录结果
        results.append({
            'sky_index': row['sky_index'],
            'ra': center.ra.deg,
            'dec': center.dec.deg,
            'az': center_altaz.az.deg,
            'alt': center_altaz.alt.deg,
            'visible': visible,
            'star_count': count,
            'weight_sum': weight,
            'fov_width': fov_width.value,
            'fov_height': fov_height.value
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # 示例WCS参数
    wcs_data = {
        'sky_index': [1, 2],
        'NAXIS1': [9576, 8192],
        'NAXIS2': [6388, 8192],
        'CTYPE1': ['RA---TAN', 'RA---TAN'],
        'CTYPE2': ['DEC--TAN', 'DEC--TAN'],
        'CRVAL1': [123.456, 185.472],
        'CRVAL2': [-45.678, 32.115],
        'CRPIX1': [4788, 4096],
        'CRPIX2': [3194, 4096],
        'CD1_1': [-0.0001, -0.00008],
        'CD1_2': [0.0, 0.00002],
        'CD2_1': [0.00003, -0.00001],
        'CD2_2': [0.0001, 0.00009]
    }

    # 运行观测系统
    results = observe_system(
        obs_time="2024-12-25 08:30:00",
        obs_lon=-116,
        obs_lat=32,
        wcs_df=pd.DataFrame(wcs_data),
        mag_limit=19,
        buffer=0.3
    )

    # 输出结果分析
    print(f"找到{len(results)}个可见天区")
    print(results[['sky_index', 'ra', 'dec', 'star_count', 'weight_sum']])
