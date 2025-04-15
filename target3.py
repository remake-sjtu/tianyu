import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.utils.iers import conf


def calculate_region_visibility(
    center_ra: u.Quantity,
    center_dec: u.Quantity,
    xfov: u.Quantity,
    yfov: u.Quantity,
    obs_time: Time,
    obs_location: EarthLocation,
    stars_icrs: SkyCoord,
    mags: np.ndarray,
) -> tuple[bool, int, float]:
    """
    计算目标天区的可见性及亮星影响
    

    center_ra: 区域中心赤经 (例: 150.5*u.deg)
    center_dec: 区域中心赤纬
    xfov: 横向视场 (例: 8*u.deg)
    yfov: 纵向视场
    obs_time: 观测时间 (Time对象)
    obs_location: 观测站位置
    stars_icrs: 星表坐标 (ICRS框架)
    mags: 星等数组
    mag_limit: 纳入计算的星等阈值
    
    
    (区域是否可见, 亮星数量, 总影响值)
    """
    # 检查目标区域自身可见性
    target_center = SkyCoord(ra=center_ra, dec=center_dec)
    target_altaz = target_center.transform_to(AltAz(
        location=obs_location,
        obstime=obs_time
    ))
    
    # 如果中心点低于地平线，直接返回不可见
    if target_altaz.alt < 0*u.deg:
        return False, 0, 0.0

    # 步骤2：筛选可见亮星
    stars_altaz = stars_icrs.transform_to(AltAz(
        location=obs_location,
        obstime=obs_time
    ))
    visible_stars = (stars_altaz.alt > 0*u.deg) 
    
    if not np.any(visible_stars):
        return True, 0, 0.0

    # 转换到目标区域的切平面坐标系
    # 将天球上的点投影到与球面某点相切的平面上，小视场近似。
    gnomonic_frame = target_center.skyoffset_frame()
    # 筛选出可见星，进行坐标转换
    stars_gnomonic = stars_icrs[visible_stars].transform_to(gnomonic_frame)
    
    # 判断是否在视场内
    x = stars_gnomonic.lon.to_value(u.deg)
    y = stars_gnomonic.lat.to_value(u.deg)
    in_fov = (
        (np.abs(x) <= xfov.to_value(u.deg)/2) &
        (np.abs(y) <= yfov.to_value(u.deg)/2)
    )
    
    # 步骤5：计算影响权重（权重与亮度呈指数关系）
    valid_mags = mags[visible_stars][in_fov]
    weights = 10**(-0.4 * (valid_mags))  # 星等每亮1等，权重增2.5倍
    
    return True, np.sum(in_fov), np.sum(weights)

# 示例使用
def main():
    # 初始化观测参数
    obs_location = EarthLocation(lon=124.4*u.deg, lat=44.9*u.deg)  
    obs_time = Time('2024-06-15 18:00:00')  
    
    # 目标区域参数
    target_ra = 50.5 * u.deg    
    target_dec = 40.2 * u.deg
    fov_x = 6 * u.deg            
    fov_y = 6 * u.deg
    
    # 生成模拟星表（包含盖亚星表部分特征）
    np.random.seed(42)
    num_stars = 100000000
    stars = SkyCoord(
        ra=np.random.uniform(0, 360, num_stars)*u.deg,
        dec=np.random.uniform(-90, 90, num_stars)*u.deg
    )
    magnitudes = np.random.normal(loc=8, scale=2, size=num_stars)
    
    # 执行计算
    visible, count, impact = calculate_region_visibility(
        center_ra=target_ra,
        center_dec=target_dec,
        xfov=fov_x,
        yfov=fov_y,
        obs_time=obs_time,
        obs_location=obs_location,
        stars_icrs=stars,
        mags=magnitudes,
    )
    
    # 输出结果
    print(f"[目标区域可见性] {'可见' if visible else '不可见'}")
    if visible:
        print(f"区域内亮星数量: {count}")
        print(f"综合影响指数: {impact:.2f} (指数越高干扰越大)")

if __name__ == "__main__":
    main()
