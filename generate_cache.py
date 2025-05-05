import os
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io.votable import parse
from scipy.spatial import KDTree

# ====================== 数据处理模块 ======================
class GaiaDataProcessor:
    """天文数据处理与缓存生成系统"""

    def __init__(self, votable_path, wcs_csv, cache_dir="gaia_cache", shards=10):
        self.votable_path = Path(votable_path)
        self.wcs_df = pd.read_csv(wcs_csv)
        self.cache_dir = Path(cache_dir)
        self.shards = shards
        os.makedirs(self.cache_dir, exist_ok=True)

        # 加载基础数据并构建空间索引
        self.data, self.coords = self._load_gaia_data()
        self.kdtree = KDTree(self.coords)
        print(f"KD-Tree initialized with {len(self.data)} stars")

    def _load_gaia_data(self):
        """加载并转换数据"""
        cache_file = self.votable_path.with_suffix('.npz')
        if cache_file.exists():
            return self._load_cached_data(cache_file)
        return self._process_votable(cache_file)

    def _load_cached_data(self, cache_file):
        """加载缓存数据"""

        npz = np.load(cache_file)
        with open(cache_file.with_suffix('.kdtree'), 'rb') as f:
            self.kdtree = pickle.load(f)  # 加载KD-Tree
        print(f"Loaded cached data from {cache_file}")
        return npz['data'], npz['coords']

    def _process_votable(self, cache_file):
        """处理VOTable文件"""
        print(f"Processing {self.votable_path}...")
        start = time.time()

        table = parse(self.votable_path).get_first_table()
        original_data = table.array

        # 创建仅包含必要字段的结构化数组
        dtype = [
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('phot_g_mean_mag', 'f4')
        ]
        data = np.empty(original_data.shape, dtype=dtype)
        data['ra'] = original_data['ra']
        data['dec'] = original_data['dec']
        data['phot_g_mean_mag'] = original_data['phot_g_mean_mag']

        coords = self.radec_to_xyz(data['ra'], data['dec'])

        np.savez(cache_file, data=data, coords=coords)
        print(f"Saved processed data to {cache_file} ({time.time()-start:.1f}s)")

        # 保存三维坐标和KD-Tree
        np.savez(cache_file, data=data, coords=coords)
        with open(cache_file.with_suffix('.kdtree'), 'wb') as f:
            pickle.dump(KDTree(coords), f)  # 保存KD-Tree对象

        return data, coords

    @staticmethod
    def radec_to_xyz(ra, dec):
        """坐标转换"""
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
        return np.column_stack((
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad)
        ))

    def process_all_regions(self, pixel_scale=3.76e-6, max_mag=12.0):
        """处理所有天区"""
        for idx, row in self.wcs_df.iterrows():
            self._process_single_region(row, pixel_scale, max_mag, idx+1)

    def _process_single_region(self, wcs_row, pixel_scale, max_mag, sky_index):
        """处理单个天区"""

        naxis1, naxis2 = wcs_row['NAXIS1'], wcs_row['NAXIS2']
        radius = self._calculate_radius(naxis1, naxis2, pixel_scale)

        stars = self._query_region(wcs_row['CRVAL1'], wcs_row['CRVAL2'], radius, max_mag)

        self._save_to_shard(wcs_row['CRVAL1'], wcs_row['CRVAL2'], radius, stars)

        if sky_index % 100 == 1 :
            print(f"SkyRegion {sky_index}: Cached {len(stars)} stars")

    def _calculate_radius(self, naxis1, naxis2, pixel_scale):
        """计算查询半径"""
        half_w = naxis1/2 * pixel_scale
        half_h = naxis2/2 * pixel_scale
        return np.sqrt(half_w**2 + half_h**2) * 1.1

    def _query_region(self, ra, dec, radius, max_mag):
        """执行区域查询"""
        search_radius = 2 * np.sin(np.deg2rad(radius) / 2)
        center = self.radec_to_xyz([ra], [dec])[0]

        indices = self.kdtree.query_ball_point(center, search_radius)
        results = self.data[indices]

        mask = results['phot_g_mean_mag'] < max_mag
        return results[mask][['ra', 'dec', 'phot_g_mean_mag']]

    def _get_shard_id(self, ra, dec):
        """分片策略"""
        return hash(f"{ra:.3f}_{dec:.3f}") % self.shards

    def _save_to_shard(self, ra, dec, radius, data):
        """保存分片数据"""
        shard_id = self._get_shard_id(ra, dec)
        shard_file = self.cache_dir / f"shard_{shard_id:02d}.npz"
        cache_key = f"{ra:.3f}_{dec:.3f}_{radius:.3f}"

        existing = dict(np.load(shard_file)) if shard_file.exists() else {}
        existing[cache_key] = data
        np.savez_compressed(shard_file, **existing)

# ====================== 查询服务模块 ======================
class GaiaQueryEngine:
    """天文数据查询引擎 """

    def __init__(self, wcs_csv, cache_dir="gaia_cache", shards=10):
        self.wcs_df = pd.read_csv(wcs_csv)
        self.cache_dir = Path(cache_dir)
        self.shards = shards
        self._verify_cache_structure()

    def _verify_cache_structure(self):
        for shard_id in range(self.shards):
            if not (self.cache_dir / f"shard_{shard_id:02d}.npz").exists():
                raise FileNotFoundError(f"Missing shard file: shard_{shard_id:02d}.npz")

    def query_by_coords(self, ra, dec, radius):
        """坐标查询"""
        raw_data = self._load_raw_data(ra, dec, radius)
        return pd.DataFrame({
            'ra_deg': raw_data['ra'],
            'dec_deg': raw_data['dec'],
            'magnitude': raw_data['phot_g_mean_mag']
        })

    def query_by_skyindex(self, sky_index):
        """天区编号查询"""
        wcs_row = self.wcs_df[self.wcs_df['sky_index'] == sky_index].iloc[0]
        radius = self._calculate_radius(
            wcs_row['NAXIS1'],
            wcs_row['NAXIS2'],
            self._parse_pixel_scale(wcs_row)
        )
        return self.query_by_coords(wcs_row['CRVAL1'], wcs_row['CRVAL2'], radius)

    def _load_raw_data(self, ra, dec, radius):
        """加载原始数据"""
        shard_id = self._get_shard_id(ra, dec)
        shard_file = self.cache_dir / f"shard_{shard_id:02d}.npz"
        cache_key = f"{ra:.3f}_{dec:.3f}_{radius:.3f}"

        try:
            data = np.load(shard_file, allow_pickle=True)
            return data[cache_key]
        except KeyError:
            raise ValueError(f"Data not found for {cache_key}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Shard file {shard_file} missing")

    def _calculate_radius(self, *args):
        return GaiaDataProcessor._calculate_radius(None, *args)

    def _parse_pixel_scale(self, wcs_row):
        cd_matrix = np.array([
            [wcs_row['CD1_1'], wcs_row['CD1_2']],
            [wcs_row['CD2_1'], wcs_row['CD2_2']]
        ])
        return np.sqrt(np.linalg.det(cd_matrix))

    def _get_shard_id(self, ra, dec):
        return GaiaDataProcessor._get_shard_id(None, ra, dec)

# ====================== 使用示例 ======================

if __name__ == "__main__":
    # 数据处理流程
    processor = GaiaDataProcessor(
        votable_path="D:/tianyu/1745858175153O-result.vot", # 星表数据文件
        wcs_csv="sky_WCS.csv" # 全量的天区的wcs信息
    )
    processor.process_all_regions(
        pixel_scale=3.76 / 2563000 * 180 / np.pi,
        max_mag=12.0
    )

    # 数据查询示例
    query_engine = GaiaQueryEngine(wcs_csv="sky_WCS.csv")

    # 示例1：天区编号查询
    print("\n示例1：天区42的恒星数据")
    stars_df = query_engine.query_by_skyindex(42)
    print(stars_df.head())

    # 示例2：直接坐标查询
    print("\n示例2：坐标(123.456, 45.678)附近恒星")
    sample_df = query_engine.query_by_coords(
        ra=123.456,
        dec=45.678,
        radius=0.5
    )
    print(f"找到{len(sample_df)}颗恒星，示例数据：")
    print(sample_df.sample(3, random_state=42))
