from astroquery.gaia import Gaia
import numpy as np


def query_gaia_stars(center_ra, center_dec, fov_width, fov_height, buffer=0.2):
    """
    :param center_ra: 图像中心赤经 (度)
    :param center_dec: 图像中心赤纬 (度)
    :param fov_width: 视场宽度 (度)
    :param fov_height: 视场高度 (度)
    :param buffer: 安全缓冲系数（建议0.2-0.3）
    :return: 包含恒星数据的Astropy Table
    """
    # 计算搜索半径（覆盖整个FOV对角线）
    radius = np.sqrt((fov_width / 2) ** 2 + (fov_height / 2) ** 2) * (1 + buffer)

    # 构建ADQL查询
    query = f"""
    SELECT 
      source_id, ra, dec, 
      phot_g_mean_mag as magnitude,
      pmra, pmdec,
      phot_bp_mean_mag, phot_rp_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 
      1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {center_ra}, {center_dec}, {radius})
      )
      AND phot_g_mean_mag < 18  -- 过滤暗星提升性能
      AND astrometric_params_solved > 0  -- 基础数据质量过滤
    """

    # 执行查询（自动分页）
    job = Gaia.launch_job_async(query)
    return job.get_results()


# 使用示例
if __name__ == "__main__":
    stars_table = query_gaia_stars(
        center_ra=123.456,
        center_dec=-45.678,
        fov_width=1.2,
        fov_height=0.8,
        buffer=0.3
    )

    stars_df = stars_table.to_pandas()
    print(f"找到 {len(stars_df)} 颗恒星")
