import pickle
from bisect import insort

import pandas as pd
import numpy as np
from pathlib import Path
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body
import astropy.units as u
from astropy.wcs.utils import wcs_to_celestial_frame, proj_plane_pixel_scales
from scipy.spatial import KDTree

from generate_cache import GaiaDataProcessor


class GaiaQueryEngine:
    """天文数据查询引擎"""

    def __init__(self, votable_path, cache_dir="gaia_cache", shards=10):
        self.votable_path = Path(votable_path)
        self.cache_dir = Path(cache_dir)
        self.shards = shards
        self._verify_cache_structure()

        # 确保缓存目录存在
        self.cache_dir.mkdir(exist_ok=True)

        # 动态加载或构建 KD-Tree
        self._prepare_kdtree()

        self._verify_cache_structure()

    def _prepare_kdtree(self):
        """动态准备 KD-Tree（无缓存时自动构建）"""
        kdtree_file = self.votable_path.with_suffix('.kdtree')
        npz_file = self.votable_path.with_suffix('.npz')

        try:
            # 尝试加载已有数据
            npz = np.load(npz_file)
            self.data = npz['data']
            self.coords = npz['coords']

            with open(kdtree_file, 'rb') as f:
                self.kdtree = pickle.load(f)
        except FileNotFoundError:
            # 从原始 VOTable 文件构建
            from astropy.io.votable import parse


            # 构建并保存
            self.kdtree = KDTree(self.coords)
            with open(kdtree_file, 'wb') as f:
                pickle.dump(self.kdtree, f)


    def _verify_cache_structure(self):
            """验证缓存完整性"""
            for shard_id in range(self.shards):
                if not (self.cache_dir / f"shard_{shard_id:02d}.npz").exists():
                    raise FileNotFoundError(f"Missing shard file: shard_{shard_id:02d}.npz")

    def query_by_coords(self, ra, dec, radius_deg, max_mag=20.0):
        try:
            # 尝试从缓存获取
            return self._load_from_cache(ra, dec, radius_deg)
        except (KeyError, FileNotFoundError):
            # 实时查询并保存结果
            realtime_data = self._query_realtime(ra, dec, radius_deg, max_mag)
            return realtime_data

    def _load_from_cache(self, ra, dec, radius_deg):
        """从缓存加载数据"""
        shard_id = self._get_shard_id(ra, dec)
        cache_key = f"{ra:.3f}_{dec:.3f}_{radius_deg:.3f}"
        data = np.load(self.cache_dir / f"shard_{shard_id:02d}.npz", allow_pickle=True)
        return data[cache_key]

    def _save_to_cache(self, ra, dec, radius_deg, data):
        """保存实时查询结果到缓存"""
        shard_id = self._get_shard_id(ra, dec)
        shard_file = self.cache_dir / f"shard_{shard_id:02d}.npz"

        existing = dict(np.load(shard_file)) if shard_file.exists() else {}
        existing[f"{ra:.3f}_{dec:.3f}_{radius_deg:.3f}"] = data
        np.savez_compressed(shard_file, **existing)

    def _get_shard_id(self, ra, dec):
        """分片策略（需与数据预处理保持一致）"""
        return hash(f"{ra:.3f}_{dec:.3f}") % self.shards

    def _query_realtime(self, ra, dec, radius_deg, max_mag=14.0):
        """实时查询KD-Tree"""
        # 计算三维坐标
        print("real time query")
        search_radius = 2 * np.sin(np.deg2rad(radius_deg) / 2)
        center = GaiaDataProcessor.radec_to_xyz([ra], [dec])[0]

        # 执行球查询
        indices = self.kdtree.query_ball_point(center, search_radius)
        results = self.data[indices]

        # 过滤星等
        mask = results['phot_g_mean_mag'] < max_mag
        return results[mask][['ra', 'dec', 'phot_g_mean_mag']]


def calculate_visibility(wcs, image_shape, stars_data):
    """可见性计算"""
    try:
        # 转换坐标系
        frame = wcs_to_celestial_frame(wcs)
        stars = SkyCoord(ra=stars_data['ra'] * u.deg,
                         dec=stars_data['dec'] * u.deg)
        stars_trans = stars.transform_to(frame)

        # 坐标转换到像素
        x, y = wcs.world_to_pixel(stars_trans)

        # 计算有效区域
        valid = (x >= 0) & (x < image_shape[0]) & \
                (y >= 0) & (y < image_shape[1])

        # 计算权重
        mags = stars_data['phot_g_mean_mag']
        weights = 10 ** (-0.2 * mags[valid])
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

    # 计算太阳的反方向区域
    sun_altaz = get_sun(time).transform_to(AltAz(location=location, obstime=time))
    anti_sun = {
        'az': (sun_altaz.az + 180 * u.deg) % (360 * u.deg),
        'alt_range': (70 * u.deg, 90 * u.deg),
        'az_width': 10 * u.deg
    }

    # 计算月亮的反方向区域
    moon_altaz = get_body("moon", time).transform_to(AltAz(location=location, obstime=time))
    anti_moon = {
        'az': (moon_altaz.az + 180 * u.deg) % (360 * u.deg),
        'alt_range': (60 * u.deg, 90 * u.deg),
        'az_width': 10 * u.deg
    }

    # 返回方位角范围和高低角范围
    return [
        (
            anti_sun['az'] - anti_sun['az_width'] / 2,
            anti_sun['az'] + anti_sun['az_width'] / 2,
            *anti_sun['alt_range']
        ),
        (
            anti_moon['az'] - anti_moon['az_width'] / 2,
            anti_moon['az'] + anti_moon['az_width'] / 2,
            *anti_moon['alt_range']
        )
    ]


def is_time_suitable(obs_time, location):
    sun_alt = get_sun(obs_time).transform_to(
        AltAz(location=location, obstime=obs_time)).alt
    return -10 * u.deg <= sun_alt <= -6 * u.deg


def observe_system(votable_path, obs_time, obs_lon, obs_lat, wcs_csv_path,mag_limit=20.0, buffer=0.2, cache_shards=10):
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
    # 初始化组件
    query_engine = GaiaQueryEngine(votable_path,shards=cache_shards)
    wcs_df = pd.read_csv(wcs_csv_path)


    # 初始化天文参数
    obs_time = Time(obs_time)
    location = EarthLocation(lon=obs_lon * u.deg, lat=obs_lat * u.deg)

    # 检查观测时间有效性
    if not is_time_suitable(obs_time, location):
        return pd.DataFrame(columns=['sky_index', 'visible', 'star_count', 'weight_sum', 'ra', 'dec', 'az', 'alt'])

    # 获取目标观测区域 反太阳区 反月亮区
    target_regions = get_target_regions(obs_time, obs_lon, obs_lat)
    target_regions_deg = [(r[0].deg, r[1].deg, r[2].deg, r[3].deg) for r in target_regions]


    sorted_results = []
    i = 0
    # 处理每个WCS区域
    for _, row in wcs_df.iterrows():
        i = i + 1
        if i == 1000:
            break
        try:
            # 解析WCS参数
            wcs = WCS(naxis=2)
            wcs.wcs.ctype = [row['CTYPE1'], row['CTYPE2']]
            wcs.wcs.crval = [row['CRVAL1'], row['CRVAL2']]
            wcs.wcs.crpix = [row['CRPIX1'], row['CRPIX2']]
            wcs.wcs.cd = [[row['CD1_1'], row['CD1_2']],
                          [row['CD2_1'], row['CD2_2']]]

            # 计算视场参数
            try:
                # 获取准确中心坐标
                center = wcs.pixel_to_world(row['CRPIX1'], row['CRPIX2'])
                center_altaz = center.transform_to(AltAz(location=location, obstime=obs_time))
            except Exception as e:
                print(f"WCS解析失败: {e}")
                continue

            #检查是否在目标区域内
            if not is_in_target_regions(center_altaz, target_regions_deg):
                continue

            pixel_scale = np.sqrt(np.linalg.det(wcs.wcs.cd))
            radius_deg = np.sqrt(
                (row['NAXIS1'] / 2 * pixel_scale) ** 2 +
                (row['NAXIS2'] / 2 * pixel_scale) ** 2
            ) * (1 + buffer)

            # 获取缓存数据
            stars_data = query_engine.query_by_coords(
                ra=row['CRVAL1'],
                dec=row['CRVAL2'],
                radius_deg=radius_deg
            )

            # 过滤星等
            valid_mag = stars_data['phot_g_mean_mag'] < mag_limit
            filtered_data = stars_data[valid_mag]

            # 可见性计算
            visible, count, weight = calculate_visibility(
                wcs=wcs,
                image_shape=(row['NAXIS1'], row['NAXIS2']),
                stars_data=filtered_data
            )

            # 构造新条目
            new_entry = {
                'sky_index': row['sky_index'],
                'ra': center.ra.deg,
                'dec': center.dec.deg,
                'az': center_altaz.az.deg,
                'alt': center_altaz.alt.deg,
                'visible': visible,
                'star_count': count,
                'weight_sum': weight,
                'search_radius': radius_deg
            }

            # 智能插入并维持排序（按weight升序）
            insort(sorted_results, (weight, new_entry))

            # 维持最多10个元素
            if len(sorted_results) > 10:
                sorted_results.pop()

        except Exception as e:
            print(f"处理天区 {row['sky_index']} 失败: {str(e)}")
            continue

            # 提取最终结果
    final_results = [entry for (_, entry) in sorted_results]
    return pd.DataFrame(final_results)


# 示例用法
if __name__ == "__main__":
    results = observe_system(
        obs_time="2024-12-25 08:30:00",
        obs_lon=-116.2153,
        obs_lat=32.2541,
        wcs_csv_path="sky_WCS.csv", # 候选天区的wcs信息
        votable_path = "D:/tianyu/1745858175153O-result.vot", # 从gaia获取的星表数据的文件
        mag_limit=19,
        buffer=0.1,
        cache_shards=10
    )

    print(f"处理完成，找到 {len(results)} 个有效天区")
    print(results[['sky_index', 'ra', 'dec', 'star_count', 'weight_sum']])
