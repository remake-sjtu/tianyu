import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.time import Time
import astropy.units as u
from scipy.spatial.transform import Rotation

# --------------------------
# 核心功能函数（已修正参数名）
# --------------------------

def generate_custom_search_grid(center_ra, center_dec, xfov, yfov, theta,
                                obs_time, obs_lon, obs_lat, step=1 * u.deg):
    """生成自定义搜索网格（单位转换强化版）"""
    center_coord = SkyCoord(ra=center_ra, dec=center_dec, frame='icrs')

    # 单位预处理（转换为数值）
    xfov_deg = xfov.to_value(u.deg)
    yfov_deg = yfov.to_value(u.deg)
    step_deg = step.to_value(u.deg)

    # 生成局部笛卡尔网格（纯数值操作）
    x_steps = np.arange(-xfov_deg / 2, xfov_deg / 2 + step_deg, step_deg)
    y_steps = np.arange(-yfov_deg / 2, yfov_deg / 2 + step_deg, step_deg)
    x, y = np.meshgrid(x_steps, y_steps)

    # 应用三维旋转（数值矩阵运算）
    rot = Rotation.from_euler('z', theta, degrees=True)
    rotated = rot.apply(np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]))

    location = EarthLocation(lon=obs_lon * u.deg, lat=obs_lat * u.deg)
    time = Time(obs_time) if isinstance(obs_time, str) else obs_time

    grid_points = []
    for dx, dy, _ in rotated:
        try:
            # 计算实际偏移距离和方向角
            separation = np.sqrt(dx ** 2 + dy ** 2) * u.deg
            position_angle = np.arctan2(dy, dx) * u.radian

            new_coord = center_coord.directional_offset_by(
                position_angle=position_angle,
                separation=separation
            )

            altaz = new_coord.transform_to(AltAz(location=location, obstime=time))
            if altaz.alt > 0 * u.deg:
                grid_points.append((
                    altaz.az.wrap_at(180 * u.deg),
                    altaz.alt
                ))
        except Exception as e:
            print(f"坐标异常：偏移量({dx:.2f}, {dy:.2f}) 错误信息: {str(e)}")
            continue

    return grid_points
# --------------------------
# 其他函数保持不变
# --------------------------

def vectorized_evaluation(search_grid, stars_vectors, star_weights, fov=5*u.deg):
    """向量化权重评估"""
    az_rad = np.array([p[0].to_value(u.rad) for p in search_grid])
    alt_rad = np.array([p[1].to_value(u.rad) for p in search_grid])

    x = np.cos(alt_rad) * np.sin(az_rad)
    y = np.cos(alt_rad) * np.cos(az_rad)
    z = np.sin(alt_rad)
    grid_vectors = np.column_stack([x, y, z])

    cos_theta = np.dot(grid_vectors, stars_vectors.T)
    cos_fov_half = np.cos(fov.to_value(u.rad)/2)
    mask = cos_theta >= cos_fov_half
    return np.dot(mask.astype(float), star_weights)

# --------------------------
# 示例使用
# --------------------------

def main():
    # 用户输入参数
    input_ra = 150.5 * u.deg       # 目标区域中心赤经
    input_dec = 40.2 * u.deg       # 目标区域中心赤纬
    input_xfov = 8 * u.deg         # 横向视场范围
    input_yfov = 6 * u.deg         # 纵向视场范围
    input_theta = 30               # 区域旋转角度（相对于赤经方向）
    obs_lon = 116.4                # 观测地经度（北京）
    obs_lat = 39.9                 # 观测地纬度
    obs_time = '2024-06-15 02:00:00'  # 观测时间（UTC）
    mag_limit = 6                  # 可见星等阈值

    # 生成模拟星表数据
    np.random.seed(42)
    num_stars = 50000
    stars_icrs = SkyCoord(
        ra=np.random.uniform(0, 360, num_stars)*u.deg,
        dec=np.random.uniform(-90, 90, num_stars)*u.deg
    )
    mags = np.random.normal(loc=8, scale=2, size=num_stars)

    # 转换到观测时的地平坐标系
    location = EarthLocation(lon=obs_lon*u.deg, lat=obs_lat*u.deg)
    time = Time(obs_time)
    stars_altaz = stars_icrs.transform_to(AltAz(location=location, obstime=time))

    # 过滤可见星
    valid_mask = (mags <= mag_limit) & (stars_altaz.alt > 0*u.deg)
    valid_stars = stars_altaz[valid_mask]
    valid_mags = mags[valid_mask]

    # 预处理星数据向量
    az_rad = valid_stars.az.to_value(u.rad)
    alt_rad = valid_stars.alt.to_value(u.rad)
    x = np.cos(alt_rad) * np.sin(az_rad)
    y = np.cos(alt_rad) * np.cos(az_rad)
    z = np.sin(alt_rad)
    stars_vectors = np.column_stack([x, y, z])
    star_weights = 100**(-0.2 * valid_mags)  # 权重计算公式

    # 生成搜索网格
    search_grid = generate_custom_search_grid(
        center_ra=input_ra,
        center_dec=input_dec,
        xfov=input_xfov,
        yfov=input_yfov,
        theta=input_theta,
        obs_time=obs_time,
        obs_lon=obs_lon,
        obs_lat=obs_lat,
        step=0.5 * u.deg
    )
    print(f"有效网格点数量: {len(search_grid)}")

    # 执行评估
    if len(search_grid) > 0:
        weights = vectorized_evaluation(search_grid, stars_vectors, star_weights)
        min_idx = np.argmin(weights)
        best_point = search_grid[min_idx]
        print(f"\n最佳指向：方位角 {best_point[0].to_value(u.deg):.2f}°, "
              f"高度角 {best_point[1].to_value(u.deg):.2f}°")
        print(f"最小干扰权重：{weights[min_idx]:.4f}")
    else:
        print("没有生成有效网格点")

if __name__ == "__main__":
    main()
