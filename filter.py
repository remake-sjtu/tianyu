import numpy as np
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
from astropy.time import Time

from target import get_target_regions

"""
    对星星进行过滤
    因为我们已经知道要搜索的区域 反太阳区域 反月亮区域，以及望远镜的fov，那么大部分星星是一定在视场外的，所以我们可以对这些星星进行过滤
    
    todo， may be可以在 从数据库中 获取 星星数据的时候就进行一个粗糙的过滤？
"""


"""
    分步过滤每个目标区域的扩展范围（支持星等过滤）

    参数：
    - stars_altaz: SkyCoord数组，恒星的AltAz坐标
    - target_regions: 目标区域列表，每个区域为(min_az, max_az, min_alt, max_alt)
    - fov_radius: 视场半径（单位：角度）
    - mags: 星等数组 todo，传入参数待定
    - mag_limit: 星等阈值（保留mag <= mag_limit的恒星）

    - valid_indices: 有效星星的索引数组
"""
def filter_stars_by_regions(stars_altaz, target_regions, fov_radius=5 * u.deg, mags=None, mag_limit=6):
    # 预处理：转换为numpy数组并应用星等过滤
    az_deg = stars_altaz.az.wrap_at(360 * u.deg).to_value(u.deg)
    alt_deg = stars_altaz.alt.to_value(u.deg)

    if mags is not None:
        bright_mask = mags <= mag_limit
        az_deg = az_deg[bright_mask]
        alt_deg = alt_deg[bright_mask]
        original_indices = np.where(bright_mask)[0]
    else:
        original_indices = np.arange(len(stars_altaz))

    # 初始化有效掩码
    valid_mask = np.zeros_like(az_deg, dtype=bool)

    # 处理每个目标区域
    for region in target_regions:
        # 计算当前区域的扩展范围
        az_intervals, (alt_min, alt_max) = calculate_single_region(region, fov_radius)

        # 计算当前区域的过滤结果
        region_mask = create_region_mask(az_deg, alt_deg, az_intervals, alt_min, alt_max)

        # 合并过滤结果
        valid_mask |= region_mask

    # 获取有效索引（映射回原始数据）
    filtered_indices = original_indices[valid_mask]
    return filtered_indices

"""计算单个区域的扩展范围"""
def calculate_single_region(region, fov_radius):
    fov_deg = fov_radius.to_value(u.deg)

    # 解析区域参数
    min_az = region[0].to_value(u.deg) if hasattr(region[0], 'to_value') else region[0]
    max_az = region[1].to_value(u.deg) if hasattr(region[1], 'to_value') else region[1]
    min_alt = region[2].to_value(u.deg) if hasattr(region[2], 'to_value') else region[2]
    max_alt = region[3].to_value(u.deg) if hasattr(region[3], 'to_value') else region[3]

    # 扩展方位角范围
    expanded_min_az = (min_az - fov_deg) % 360
    expanded_max_az = (max_az + fov_deg) % 360

    # 生成方位角区间
    if expanded_min_az <= expanded_max_az:
        az_intervals = [(expanded_min_az, expanded_max_az)]
    else:
        az_intervals = [(expanded_min_az, 360.0), (0.0, expanded_max_az)]

    # 扩展高度角范围（0-90度）
    expanded_min_alt = max(0.0, min_alt - fov_deg)
    expanded_max_alt = min(90.0, max_alt + fov_deg)

    return az_intervals, (expanded_min_alt, expanded_max_alt)


def create_region_mask(az_deg, alt_deg, az_intervals, alt_min, alt_max):
    """计算当前区域的过滤结果"""
    # 高度角过滤
    alt_mask = (alt_deg >= alt_min) & (alt_deg <= alt_max)

    # 方位角过滤
    az_mask = np.zeros_like(az_deg, dtype=bool)
    for a_min, a_max in az_intervals:
        if a_min <= a_max:
            az_mask |= (az_deg >= a_min) & (az_deg <= a_max)
        else:
            az_mask |= (az_deg >= a_min) | (az_deg <= a_max)

    return alt_mask & az_mask


# 示例用法
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    # 大概需要1-2分钟计算
    num_stars = 100000000
    stars_icrs = SkyCoord(
        ra=np.random.uniform(0, 360, num_stars) * u.deg,
        dec=np.random.uniform(-90, 90, num_stars) * u.deg
    )
    mags = np.random.uniform(0, 10, num_stars)

    # 转换为地平坐标
    location = EarthLocation(lon=111 * u.deg, lat=22 * u.deg)
    time = Time('2024-06-06 06:06:06')
    stars_altaz = stars_icrs.transform_to(AltAz(location=location, obstime=time))

    # 获取目标区域
    target_regions = get_target_regions(time, 111, 22)  # 使用之前定义的get_target_regions

    # 执行分步过滤
    valid_indices = filter_stars_by_regions(
        stars_altaz,
        target_regions,
        fov_radius=5 * u.deg,
        mags=mags,
        mag_limit=6
    )

    print(f"原始星数: {len(stars_altaz)}")
    print(f"过滤后星数: {len(valid_indices)}")
    print(f"过滤率: {len(valid_indices) / len(stars_altaz):.2%}")
