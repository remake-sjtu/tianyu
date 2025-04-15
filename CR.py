import pandas as pd
import numpy as np

# ====================
# 数据生成参数配置
# ====================
provinces = [
    "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省",
    "河南省", "湖北省", "湖南省", "广东省", "广西省", "四川省",
    "黑龙江省", "辽宁省", "河北省"
]

# 基准参数 (省份: [基准收入(亿), 初始水域面积, 初始劳动力])
base_params = {
    "江苏省": [800, 680, 420],
    "浙江省": [700, 550, 380],
    "安徽省": [600, 450, 320],
    "福建省": [650, 500, 300],
    "江西省": [580, 480, 280],
    "山东省": [750, 600, 350],
    "河南省": [550, 400, 300],
    "湖北省": [620, 520, 310],
    "湖南省": [600, 500, 290],
    "广东省": [720, 650, 400],
    "广西省": [530, 420, 270],
    "四川省": [670, 550, 330],
    "黑龙江省": [480, 380, 250],
    "辽宁省": [570, 430, 260],
    "河北省": [590, 440, 275]
}


# ====================
# 核心数据生成函数
# ====================
def generate_panel_data():
    np.random.seed(2023)  # 固定随机种子保证可复现

    data = []
    for province in provinces:
        base_income, water_area, labor = base_params[province]

        # 生成逐年数据
        for year in range(2013, 2023):
            # 时间趋势
            t = year - 2013

            # 水域面积变化 (每年随机增长0-1%)
            water_area *= 1 + np.random.uniform(0, 0.01)

            # 劳动力变化 (每年减少0.5%±0.2%)
            labor *= 1 - (0.005 + np.random.uniform(-0.002, 0.002))

            # 气温生成 (与水域面积负相关)
            temp = 15 + 0.1 * t - 0.004 * (water_area - 400) + np.random.normal(0, 0.3)

            # 降水量生成 (基准+时间趋势+随机)
            precip = 800 + 30 * t + np.random.normal(0, 50)

            # 政策虚拟变量 (华东六省从2019年起为1)
            policy = 1 if (year >= 2019) and (province in ["江苏省", "浙江省", "安徽省",
                                                           "福建省", "江西省", "山东省"]) else 0

            # 养殖收入计算
            income = (
                    base_income * (1.05) ** t +  # 年增长基线
                    2.3 * precip -  # 降水效应
                    1.8 * temp +  # 气温效应
                    0.6 * water_area +  # 水域面积效应
                    80 * policy +  # 政策效应
                    np.random.normal(0, 15)  # 随机扰动
            )

            data.append([
                province, year, round(temp, 1), int(precip),
                round(income, 1), int(water_area), int(labor), policy
            ])

    return pd.DataFrame(
        data,
        columns=["省份", "年份", "气温(℃)", "降水(mm)",
                 "养殖收入(亿元)", "水域面积(千公顷)", "劳动力(万人)", "政策虚拟变量"]
    )


if __name__ == "__main__":
    # ====================
    # 生成并保存数据
    # ====================
    df = generate_panel_data()

    # 查看数据结构
    print(f"数据维度: {df.shape}")
    print(df.head(10))

    # 导出为CSV
    df.to_csv("province_aquaculture_panel.csv", index=False, encoding='utf_8_sig')

    # ====================
    # 数据验证（修正版）
    # ====================
    # 验证政策效应 (华东六省2019年前后对比)
    policy_provinces = ["江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省"]
    policy_effect = (
        df[df["省份"].isin(policy_provinces)]
        .groupby(["省份", (df["年份"] >= 2019)])["养殖收入(亿元)"]
        .mean()
        .unstack()
        .rename(columns={False: "pre_2019", True: "post_2019"})  # 重命名列
    )

    print("\n政策效应验证(华东六省平均):")
    print(round(policy_effect["post_2019"] - policy_effect["pre_2019"], 1))
