import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body
from astropy.time import Time
import astropy.units as u
from astropy.utils.iers import conf

'''
从反太阳区，反月亮区，选择一个指向（地平坐标系），该指向的视场内，星星最少

主要是将区域进行矩阵化，计算确定指向和星星向量的cos夹角，判断在该指向下，星星是否在望远镜指向的视场范围内


'''

# 观测时间 经度 纬度
def get_target_regions(obs_time, obs_lon, obs_lat):
    """获取需要搜索的目标区域（反太阳/月亮方向）"""
    location = EarthLocation(lon=obs_lon * u.deg, lat=obs_lat * u.deg)
    time = Time(obs_time)

    # 计算太阳反方向区域  Altitude  Azimuth
    # 地平坐标系 alt 高度角， az 方位角 相对于正北方向的水平角度
    sun_altaz = get_sun(time).transform_to(AltAz(location=location, obstime=time))
    anti_sun_az = (sun_altaz.az + 180 * u.deg) % (360 * u.deg)

    # 计算月亮反方向区域
    moon_altaz = get_body("moon", time).transform_to(AltAz(location=location, obstime=time))
    anti_moon_az = (moon_altaz.az + 180 * u.deg) % (360 * u.deg)

    # 方位角范围 高度角范围
    return [
        (anti_sun_az - 5 * u.deg, anti_sun_az + 5 * u.deg, 70 * u.deg, 90 * u.deg),  # 反太阳区
        (anti_moon_az - 5 * u.deg, anti_moon_az + 5 * u.deg, 60 * u.deg, 90 * u.deg)  # 反月亮区
    ]


def generate_search_grid(target_regions, step=1 * u.deg):
    """在目标区域内生成搜索网格"""
    grid_points = []
    step_deg = step.to_value(u.deg)  # 转换为纯数值

    for region in target_regions:
        # 提取数值并转换为度数
        min_az = region[0].to_value(u.deg)
        max_az = region[1].to_value(u.deg)
        min_alt = region[2].to_value(u.deg)
        max_alt = region[3].to_value(u.deg)

        # 方位角环状分布
        if max_az < min_az:
            az_values = np.concatenate([
                np.arange(min_az, 360, step_deg),
                np.arange(0, max_az + step_deg, step_deg)
            ])
        else:
            az_values = np.arange(min_az, max_az + step_deg, step_deg)

        # 生成高度角范围
        alt_values = np.arange(min_alt, max_alt + step_deg, step_deg)

        # 创建网格并添加单位
        az_grid, alt_grid = np.meshgrid(az_values, alt_values)
        grid_points.extend(zip(az_grid.ravel() * u.deg, alt_grid.ravel() * u.deg))

    return grid_points


def vectorized_evaluation(search_grid, stars_vectors, fov=5 * u.deg):
    # 将网格点转换为弧度数值
    az_rad = np.array([p[0].to_value(u.rad) for p in search_grid])
    alt_rad = np.array([p[1].to_value(u.rad) for p in search_grid])

    # 转换为单位向量，球坐标系到笛卡尔坐标系
    x = np.cos(alt_rad) * np.sin(az_rad)
    y = np.cos(alt_rad) * np.cos(az_rad)
    z = np.sin(alt_rad)
    grid_vectors = np.column_stack([x, y, z])

    # 计算点积矩阵
    cos_theta = np.dot(grid_vectors, stars_vectors.T)

    # 计算FOV阈值（转换为数值比较）
    # 点积计算进行比较，也就是星星在对应网格点（指示方向）和视场极限范围之间
    cos_fov_half = np.cos(fov.to_value(u.rad) / 2)

    # 统计每个网格点可以看见的星星数量
    return np.sum(cos_theta >= cos_fov_half, axis=1)

def vectorized_evaluation(search_grid, stars_vectors, star_weights, fov=5 * u.deg):
    # 修改为权重评估
    az_rad = np.array([p[0].to_value(u.rad) for p in search_grid])
    alt_rad = np.array([p[1].to_value(u.rad) for p in search_grid])


    # 转换为单位向量，球坐标系到笛卡尔坐标系
    x = np.cos(alt_rad) * np.sin(az_rad)
    y = np.cos(alt_rad) * np.cos(az_rad)
    z = np.sin(alt_rad)
    grid_vectors = np.column_stack([x, y, z])

    # 计算点积矩阵
    cos_theta = np.dot(grid_vectors, stars_vectors.T)

    # 计算FOV阈值（转换为数值比较）
    # 点积计算进行比较，也就是星星在对应网格点（指示方向）和视场极限范围之间
    cos_fov_half = np.cos(fov.to_value(u.rad) / 2)

    mask = cos_theta >= cos_fov_half
    weighted_sums = np.dot(mask.astype(float), star_weights)  # 使用浮点矩阵乘法

    return weighted_sums


# 示例用法
if __name__ == "__main__":
    # 生成带星等的模拟数据
    np.random.seed(111)
    num_stars = 10000
    stars_icrs = SkyCoord(
        ra=np.random.uniform(0, 360, num_stars) * u.deg,
        dec=np.random.uniform(-90, 90, num_stars) * u.deg
    )
    mags = np.random.uniform(0, 10, num_stars)  # 星等范围0-10

    location = EarthLocation(lon=111 * u.deg, lat=22 * u.deg)
    time = Time('2024-06-06 6:06:66')
    stars_altaz = stars_icrs.transform_to(AltAz(location=location, obstime=time))

    # 过滤亮星（假设mag_limit=6）
    mag_limit = 6
    valid_mask = mags <= mag_limit
    valid_stars_altaz = stars_altaz[valid_mask]
    valid_mags = mags[valid_mask]

    # 预处理星数据
    az_rad = valid_stars_altaz.az.to_value(u.rad)
    alt_rad = valid_stars_altaz.alt.to_value(u.rad)

    x = np.cos(alt_rad) * np.sin(az_rad)
    y = np.cos(alt_rad) * np.cos(az_rad)
    z = np.sin(alt_rad)
    stars_vectors = np.column_stack([x, y, z])
    star_weights = 100 ** (-0.2 * valid_mags)  # 计算权重

    # 获取目标区域并生成网格
    regions = get_target_regions(time, 111, 22)
    search_grid = generate_search_grid(regions, step=0.1 * u.deg)
    print(f"生成网格点数量: {len(search_grid)}")

    # 计算权重总和
    weight_totals = vectorized_evaluation(search_grid, stars_vectors, star_weights)

    # 寻找最优解
    if len(weight_totals) > 0:
        min_index = np.argmin(weight_totals)
        best_point = search_grid[min_index]
        print(f"最佳指向：方位角 {best_point[0].to_value(u.deg):.1f}°，高度角 {best_point[1].to_value(u.deg):.1f}°")
        print(f"总干扰权重：{weight_totals[min_index]:.4f}")
    else:
        print("未找到有效网格点")