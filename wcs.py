from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import numpy as np
from astropy.wcs.utils import wcs_to_celestial_frame


def calculate_region_visibility(
        wcs: WCS,
        image_shape: tuple,  # (图像宽度, 图像高度) 单位像素
        obs_time: Time,
        obs_location: EarthLocation,
        stars_icrs: SkyCoord,
        mags: np.ndarray,
        mag_limit: float = 20.0  # 星等阈值，默认包含所有可见星
) -> tuple[bool, int, float]:
    """
    使用WCS判断星体在图像中的可见性

    wcs: WCS对象，描述天文图像的坐标系统
    image_shape: 图像尺寸 (width, height)
    obs_time: 观测时间
    obs_location: 观测站位置
    stars_icrs: 星表坐标 (ICRS框架)
    mags: 星等数组
    mag_limit: 只考虑亮于此星等的恒星
    """

    # 检查图像中心是否可见
    try:
        center_coord = wcs.pixel_to_world(image_shape[0] // 2, image_shape[1] // 2)
        center_altaz = center_coord.transform_to(AltAz(
            location=obs_location,
            obstime=obs_time
        ))
        if center_altaz.alt < 0 * u.deg:
            return (False, 0, 0.0)
    except:
        return (False, 0, 0.0)

    # 坐标转换到WCS对应的坐标系
    target_frame = wcs_to_celestial_frame(wcs)
    stars_transformed = stars_icrs.transform_to(target_frame)

    # 转换为像素坐标
    x, y = wcs.world_to_pixel(stars_transformed)

    # 判断是否在图像范围内
    in_image = (x >= 0) & (x < image_shape[0]) & (y >= 0) & (y < image_shape[1])
    bright_enough = mags <= mag_limit
    valid = in_image & bright_enough

    # 计算影响值
    weights = 10 ** (-0.4 * (mags[valid]))
    return (True, np.sum(valid), np.sum(weights))
