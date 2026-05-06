from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from openpyxl import load_workbook


COLUMN_NAMES = [
    "sample_id",
    "constitution_label",
    "score_pinghe",
    "score_qixu",
    "score_yangxu",
    "score_yinxu",
    "score_tanshi",
    "score_shire",
    "score_xueyu",
    "score_qiyu",
    "score_tebing",
    "adl_walk",
    "adl_eat",
    "adl_dress",
    "adl_toilet",
    "adl_bath",
    "adl_total",
    "iadl_shop",
    "iadl_cook",
    "iadl_finance",
    "iadl_transport",
    "iadl_medicine",
    "iadl_total",
    "activity_total",
    "hdl_c",
    "ldl_c",
    "tg",
    "tc",
    "glucose",
    "uric_acid",
    "bmi",
    "hyperlipidemia_label",
    "hyperlipidemia_type",
    "age_group",
    "gender",
    "smoking",
    "drinking",
]


LIPID_LIMITS = {
    "tc": 6.2,
    "tg": 1.7,
    "ldl_c": 3.1,
    "hdl_c": 1.04,
}


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def load_xlsx(path: Path) -> Dict[str, np.ndarray]:
    wb = load_workbook(path, data_only=True)
    ws = wb[wb.sheetnames[0]]

    values: List[List[float]] = []
    for r in range(2, ws.max_row + 1):
        row_vals = []
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if v is None:
                row_vals.append(np.nan)
            else:
                row_vals.append(float(v))
        values.append(row_vals)

    data = np.asarray(values, dtype=float)
    return {name: data[:, i] for i, name in enumerate(COLUMN_NAMES)}


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd < 1e-9:
        return np.zeros_like(x)
    return (x - mu) / sd


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def build_risk_proxy(data: Dict[str, np.ndarray]) -> np.ndarray:
    # 用题目关键指标构造可解释的连续风险代理分值，再映射到[0,1]
    x_tg = zscore(data["tg"])
    x_tc = zscore(data["tc"])
    x_ldl = zscore(data["ldl_c"])
    x_hdl = -zscore(data["hdl_c"])
    x_tanshi = zscore(data["score_tanshi"])
    x_act = -zscore(data["activity_total"])

    score = 0.27 * x_tg + 0.24 * x_tc + 0.20 * x_ldl + 0.12 * x_hdl + 0.11 * x_tanshi + 0.06 * x_act
    return sigmoid(score)


def draw_3d_scatter(data: Dict[str, np.ndarray], out: Path) -> None:
    risk = build_risk_proxy(data)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = data["score_tanshi"]
    ys = data["tg"]
    zs = data["ldl_c"]

    sc = ax.scatter(xs, ys, zs, c=risk, cmap="turbo", s=20, alpha=0.85)
    cbar = fig.colorbar(sc, pad=0.12)
    cbar.set_label("风险概率代理值")

    ax.set_xlabel("痰湿质积分")
    ax.set_ylabel("TG (mmol/L)")
    ax.set_zlabel("LDL-C (mmol/L)")
    ax.set_title("图4 三维散点图：痰湿-血脂-风险分布")
    ax.view_init(elev=22, azim=42)

    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def draw_3d_surface(data: Dict[str, np.ndarray], out: Path) -> None:
    tg = data["tg"]
    ldl = data["ldl_c"]

    tg_min, tg_max = np.nanpercentile(tg, 5), np.nanpercentile(tg, 95)
    ldl_min, ldl_max = np.nanpercentile(ldl, 5), np.nanpercentile(ldl, 95)

    gx = np.linspace(tg_min, tg_max, 60)
    gy = np.linspace(ldl_min, ldl_max, 60)
    X, Y = np.meshgrid(gx, gy)

    # 固定痰湿中位数，展示风险响应曲面
    t0 = np.nanmedian(data["score_tanshi"])
    a0 = np.nanmedian(data["activity_total"])
    h0 = np.nanmedian(data["hdl_c"])
    tc0 = np.nanmedian(data["tc"])

    score = (
        0.27 * zscore(X)
        + 0.24 * zscore(np.full_like(X, tc0))
        + 0.20 * zscore(Y)
        + 0.12 * (-zscore(np.full_like(X, h0)))
        + 0.11 * zscore(np.full_like(X, t0))
        + 0.06 * (-zscore(np.full_like(X, a0)))
    )
    Z = sigmoid(score)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    cbar = fig.colorbar(surf, pad=0.1)
    cbar.set_label("风险概率代理值")

    ax.set_xlabel("TG (mmol/L)")
    ax.set_ylabel("LDL-C (mmol/L)")
    ax.set_zlabel("风险概率")
    ax.set_title("图5 三维曲面图：血脂指标对风险的响应")
    ax.view_init(elev=28, azim=-125)

    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, fc: str = "#e8f1fb") -> None:
    rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#2f5c8a", linewidth=1.6)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)


def _arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.7, color="#1f3b5b"))


def draw_tree_risk(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    _draw_box(ax, 0.41, 0.83, 0.18, 0.11, "根节点\n总体人群")
    _draw_box(ax, 0.09, 0.57, 0.25, 0.12, "节点A\n血脂异常项 >= 2 ?")
    _draw_box(ax, 0.38, 0.57, 0.25, 0.12, "节点B\n痰湿积分 >= 60 ?")
    _draw_box(ax, 0.67, 0.57, 0.25, 0.12, "节点C\n概率 p >= 0.70 ?")

    _draw_box(ax, 0.07, 0.28, 0.22, 0.12, "叶1\n高风险", "#fde2e1")
    _draw_box(ax, 0.33, 0.28, 0.22, 0.12, "叶2\n中风险", "#fff4d6")
    _draw_box(ax, 0.59, 0.28, 0.22, 0.12, "叶3\n中风险", "#fff4d6")
    _draw_box(ax, 0.82, 0.28, 0.16, 0.12, "叶4\n低风险", "#dff3e4")

    _arrow(ax, 0.50, 0.83, 0.21, 0.69)
    _arrow(ax, 0.50, 0.83, 0.50, 0.69)
    _arrow(ax, 0.50, 0.83, 0.79, 0.69)

    _arrow(ax, 0.21, 0.57, 0.18, 0.40)
    _arrow(ax, 0.50, 0.57, 0.44, 0.40)
    _arrow(ax, 0.79, 0.57, 0.70, 0.40)
    _arrow(ax, 0.79, 0.57, 0.90, 0.40)

    ax.text(0.13, 0.48, "是", fontsize=10)
    ax.text(0.39, 0.48, "边界", fontsize=10)
    ax.text(0.64, 0.48, "否", fontsize=10)
    ax.text(0.87, 0.48, "否", fontsize=10)

    ax.set_title("图6 树状图：三级风险分层决策路径", fontsize=15)
    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def draw_tree_intervention(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    _draw_box(ax, 0.42, 0.84, 0.16, 0.10, "根节点\n痰湿患者")
    _draw_box(ax, 0.10, 0.62, 0.24, 0.11, "年龄分层\n40-59 / 60-79 / 80+")
    _draw_box(ax, 0.38, 0.62, 0.24, 0.11, "活动能力\n<40 / 40-59 / >=60")
    _draw_box(ax, 0.66, 0.62, 0.24, 0.11, "预算约束\n总成本 <= 2000")

    _draw_box(ax, 0.08, 0.36, 0.24, 0.11, "动作集A\n强度1, 频次1-10")
    _draw_box(ax, 0.36, 0.36, 0.24, 0.11, "动作集B\n强度1-2, 频次1-10")
    _draw_box(ax, 0.64, 0.36, 0.24, 0.11, "动作集C\n强度1-3, 频次1-10")

    _draw_box(ax, 0.30, 0.12, 0.40, 0.13, "叶节点\n6个月最优方案（终点积分最小，同分成本最小）", "#e3f2e7")

    _arrow(ax, 0.50, 0.84, 0.22, 0.73)
    _arrow(ax, 0.50, 0.84, 0.50, 0.73)
    _arrow(ax, 0.50, 0.84, 0.78, 0.73)

    _arrow(ax, 0.22, 0.62, 0.20, 0.47)
    _arrow(ax, 0.50, 0.62, 0.48, 0.47)
    _arrow(ax, 0.78, 0.62, 0.76, 0.47)

    _arrow(ax, 0.20, 0.36, 0.50, 0.25)
    _arrow(ax, 0.48, 0.36, 0.50, 0.25)
    _arrow(ax, 0.76, 0.36, 0.50, 0.25)

    ax.set_title("图7 树状图：个体化干预优化决策树", fontsize=15)
    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def draw_tree_indicator(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    _draw_box(ax, 0.41, 0.84, 0.18, 0.10, "根节点\n风险预警指标")

    _draw_box(ax, 0.08, 0.61, 0.24, 0.11, "一级\n体质特征")
    _draw_box(ax, 0.38, 0.61, 0.24, 0.11, "一级\n生化特征")
    _draw_box(ax, 0.68, 0.61, 0.24, 0.11, "一级\n行为特征")

    _draw_box(ax, 0.05, 0.34, 0.27, 0.12, "二级\n痰湿/气虚/阳虚/阴虚")
    _draw_box(ax, 0.35, 0.34, 0.27, 0.12, "二级\nTC/TG/LDL-C/HDL-C")
    _draw_box(ax, 0.65, 0.34, 0.27, 0.12, "二级\n活动总分/吸烟/饮酒")

    _draw_box(ax, 0.28, 0.12, 0.44, 0.12, "叶节点\n综合评分 + 规则阈值输出三级风险", "#e3f2e7")

    _arrow(ax, 0.50, 0.84, 0.20, 0.72)
    _arrow(ax, 0.50, 0.84, 0.50, 0.72)
    _arrow(ax, 0.50, 0.84, 0.80, 0.72)

    _arrow(ax, 0.20, 0.61, 0.18, 0.46)
    _arrow(ax, 0.50, 0.61, 0.48, 0.46)
    _arrow(ax, 0.80, 0.61, 0.78, 0.46)

    _arrow(ax, 0.18, 0.34, 0.50, 0.24)
    _arrow(ax, 0.48, 0.34, 0.50, 0.24)
    _arrow(ax, 0.78, 0.34, 0.50, 0.24)

    ax.set_title("图8 树状图：多维指标体系层级结构", fontsize=15)
    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def draw_correlation_heatmap(data: Dict[str, np.ndarray], out: Path) -> None:
    names = ["score_tanshi", "activity_total", "tc", "tg", "ldl_c", "hdl_c", "uric_acid", "bmi"]
    labels = ["痰湿", "活动", "TC", "TG", "LDL-C", "HDL-C", "尿酸", "BMI"]
    X = np.column_stack([data[n] for n in names])
    corr = np.corrcoef(X, rowvar=False)

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    cbar = fig.colorbar(im)
    cbar.set_label("相关系数")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title("图9 核心指标相关性热力图")
    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def draw_literature_threshold_chart(out: Path) -> None:
    # 文献启发示意图：不同血脂指标阈值区间
    metrics = ["TC", "TG", "LDL-C", "HDL-C(低值风险)"]
    normal = [5.2, 1.7, 3.4, 1.0]
    borderline = [6.2, 2.3, 4.1, 0.9]
    high = [7.2, 5.6, 4.9, 0.8]

    x = np.arange(len(metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, normal, width=width, label="理想/建议阈值", color="#9ad0c2")
    ax.bar(x, borderline, width=width, label="临界区间上界", color="#f7c873")
    ax.bar(x + width, high, width=width, label="高风险区间参考", color="#ef8a62")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("mmol/L")
    ax.set_title("图10 文献启发示意：血脂阈值分层对照")
    ax.legend()
    ax.grid(axis="y", alpha=0.28)

    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def draw_literature_ldl_benefit(out: Path) -> None:
    # 文献启发示意图：LDL-C下降与相对风险下降的近似关系
    ldl_drop = np.array([0.2, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0])
    # 按经验近似：每降低1 mmol/L，事件风险可下降约20%左右
    risk_reduction = 1.0 - np.exp(-0.22 * ldl_drop)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(ldl_drop, risk_reduction * 100, marker="o", lw=2.6, color="#2a6f97")
    ax.fill_between(ldl_drop, risk_reduction * 100, alpha=0.2, color="#61a5c2")

    for x, y in zip(ldl_drop, risk_reduction * 100):
        ax.text(x, y + 0.8, f"{y:.1f}%", ha="center", fontsize=9)

    ax.set_xlabel("LDL-C下降幅度 (mmol/L)")
    ax.set_ylabel("相对风险下降 (%)")
    ax.set_title("图11 文献启发示意：LDL-C降低与风险获益")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def draw_literature_management_loop(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.axis("off")

    nodes = [
        (0.08, 0.40, 0.18, 0.16, "筛查评估\n(体质+血脂+行为)"),
        (0.31, 0.66, 0.18, 0.16, "风险分层\n(低/中/高)"),
        (0.54, 0.40, 0.18, 0.16, "干预执行\n(饮食+运动+调理)"),
        (0.31, 0.14, 0.18, 0.16, "随访复评\n(月度滚动)")
    ]

    for x, y, w, h, text in nodes:
        _draw_box(ax, x, y, w, h, text, fc="#eef7ff")

    _arrow(ax, 0.26, 0.48, 0.31, 0.74)
    _arrow(ax, 0.49, 0.74, 0.54, 0.48)
    _arrow(ax, 0.63, 0.40, 0.43, 0.30)
    _arrow(ax, 0.31, 0.22, 0.17, 0.40)

    ax.text(0.73, 0.52, "文献共识：\n持续管理优于单次干预", fontsize=11, color="#1d3557")
    ax.set_title("图12 文献启发示意：慢病管理闭环流程", fontsize=15)

    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def draw_all(data_path: Path, images_dir: Path) -> None:
    set_chinese_font()
    images_dir.mkdir(parents=True, exist_ok=True)
    data = load_xlsx(data_path)

    draw_3d_scatter(data, images_dir / "图4_三维散点_痰湿与血脂风险.png")
    draw_3d_surface(data, images_dir / "图5_三维曲面_风险概率响应.png")

    draw_tree_risk(images_dir / "图6_树状图_三级风险决策路径.png")
    draw_tree_intervention(images_dir / "图7_树状图_个体化干预决策树.png")
    draw_tree_indicator(images_dir / "图8_树状图_指标体系层级结构.png")

    draw_correlation_heatmap(data, images_dir / "图9_相关性热力图_核心指标.png")

    draw_literature_threshold_chart(images_dir / "图10_文献启发_血脂阈值分层示意.png")
    draw_literature_ldl_benefit(images_dir / "图11_文献启发_LDL下降与风险获益.png")
    draw_literature_management_loop(images_dir / "图12_文献启发_慢病管理闭环流程.png")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    draw_all(base.parent / "附件1：样例数据.xlsx", base / "images")
    print("参考数据图片已生成完成。")
