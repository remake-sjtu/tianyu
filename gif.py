
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np


def main_process():
    history = []

    np.random.seed(42)
    current_best = {'ra': 0, 'dec': 0, 'score': float('inf')}  # 初始化最佳记录

    for i in range(50):
        candidates = pd.DataFrame({
            'ra': np.random.uniform(0, 360, 20),
            'dec': np.random.uniform(-90, 90, 20),
            'score': np.random.exponential(1, 20)
        })

        # 过滤无效坐标
        candidates = candidates[
            (candidates['dec'].between(-90, 90)) &
            (candidates['ra'].between(0, 360))
            ]

        if not candidates.empty:
            current_candidate = candidates.loc[candidates['score'].idxmin()]
            if current_candidate['score'] < current_best['score']:
                current_best = current_candidate.copy()

            history.append({
                'iteration': i,
                'current_ra': current_candidate['ra'],
                'current_dec': current_candidate['dec'],
                'best_ra': current_best['ra'],
                'best_dec': current_best['dec'],
                'score': current_candidate['score'],
                'best_score': current_best['score']
            })

    # 确保至少有一个记录
    if not history:
        raise ValueError("主流程未能生成有效数据")

    pd.DataFrame(history).to_csv('observation_history.csv', index=False)


def create_animation(input_file='observation_history.csv', output_gif='observation_evolution.gif'):
    # 读取历史数据
    history = pd.read_csv(input_file)

    # 设置画布
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="aitoff")

    current_point = ax.scatter([], [], c='red', s=50, alpha=0.8, label='Current Candidate', zorder=3)
    best_point = ax.scatter([], [], c='lime', s=100, marker='*', alpha=0.8, label='Best Area', zorder=4)

    # 文本信息布局优化
    info_box = {
        'x': 0.98,  # 右侧对齐
        'y0': 0.95,  # 顶部起始位置
        'dy': -0.06,  # 行间距
        'fontsize': 9,
        'ha': 'right',  # 右对齐
        'va': 'top',  # 顶部对齐
        'bbox': dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none')
    }

    iter_text = ax.text(info_box['x'], info_box['y0'],
                        '',
                        transform=ax.transAxes,
                        color='white',
                        fontsize=info_box['fontsize'],
                        ha=info_box['ha'],
                        va=info_box['va'],
                        bbox=info_box['bbox'])

    coord_text = ax.text(info_box['x'], info_box['y0'] + info_box['dy'],
                         '',
                         transform=ax.transAxes,
                         color='cyan',
                         fontsize=info_box['fontsize'],
                         ha=info_box['ha'],
                         va=info_box['va'])

    score_text = ax.text(info_box['x'], info_box['y0'] + 2 * info_box['dy'],
                         '',
                         transform=ax.transAxes,
                         color='yellow',
                         fontsize=info_box['fontsize'],
                         ha=info_box['ha'],
                         va=info_box['va'])
    # 绘制银河系平面（优化坐标转换）
    def plot_galactic_plane():
        try:
            l = np.linspace(-180, 180, 500)
            b = np.zeros_like(l)
            gc = SkyCoord(l=l * u.deg, b=b * u.deg, frame='galactic')
            eq = gc.transform_to('icrs')

            # 转换到投影坐标
            ra_rad = np.radians(eq.ra.wrap_at(180 * u.deg).deg)
            dec_rad = np.radians(eq.dec.deg)

            # 分割连续线段
            split_idx = np.where(np.abs(np.diff(ra_rad)) > np.pi)[0] + 1
            for segment in np.split(np.column_stack([ra_rad, dec_rad]), split_idx):
                if len(segment) > 1:
                    ax.plot(segment[:, 0], segment[:, 1],
                            color='dodgerblue', alpha=0.3,
                            linestyle='--', zorder=1)
        except Exception as e:
            print(f"银河系平面绘制异常: {str(e)}")

    plot_galactic_plane()

    # 坐标标签设置
    ax.set_xticklabels(['14h', '16h', '18h', '20h', '22h', '0h', '2h', '4h', '6h', '8h', '10h'])
    ax.tick_params(axis='both', colors='white', labelsize=8)
    ax.grid(color='lightgray', linestyle=':', alpha=0.3, zorder=0)

    # 动态更新函数
    def update(frame):
        data = history.iloc[frame]

        # 坐标转换（处理RA环绕）
        current_ra = (data['current_ra'] + 180) % 360 - 180
        current_ra_rad = np.radians(current_ra)
        current_dec_rad = np.radians(data['current_dec'])

        best_ra = (data['best_ra'] + 180) % 360 - 180
        best_ra_rad = np.radians(best_ra)
        best_dec_rad = np.radians(data['best_dec'])

        # 更新当前点
        current_point.set_offsets(np.c_[current_ra_rad, current_dec_rad])

        # 更新最佳点（保留历史轨迹）
        if frame == 0 or data['best_score'] < history.iloc[:frame]['best_score'].min():
            best_point.set_offsets(np.c_[best_ra_rad, best_dec_rad])

        # 更新文本
        iter_text.set_text(f"Iteration: {frame + 1}/{len(history)}")

        coord_str = (f"RA: {data['current_ra']:.1f}°\n"  
                     f"Dec: {data['current_dec']:.1f}°")
        coord_text.set_text(coord_str)

        score_str = (f"Current: {data['score']:.2f}\n"
                     f"Best: {data['best_score']:.2f}")
        score_text.set_text(score_str)

        return current_point, best_point, iter_text, coord_text, score_text
    # 创建动画
    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(history),
        interval=300,
        blit=True,
        repeat_delay=1500
    )

    # 添加图例
    ax.legend(loc='lower right', fontsize=8, framealpha=0.3)

    # 保存动画
    ani.save(output_gif, writer='pillow', dpi=150,
             progress_callback=lambda i, n: print(f"\r生成进度: {i + 1}/{n}", end=""))
    print(f"\n动画已保存至: {output_gif}")

    return ani



if __name__ == "__main__":
    # 生成测试数据
    main_process()

    # 创建动画
    create_animation()
