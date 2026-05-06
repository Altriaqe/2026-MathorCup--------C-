from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from openpyxl import Workbook, load_workbook


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


DISPLAY_NAMES = {
	"sample_id": "样本ID",
	"constitution_label": "体质标签",
	"score_pinghe": "平和质",
	"score_qixu": "气虚质",
	"score_yangxu": "阳虚质",
	"score_yinxu": "阴虚质",
	"score_tanshi": "痰湿质",
	"score_shire": "湿热质",
	"score_xueyu": "血瘀质",
	"score_qiyu": "气郁质",
	"score_tebing": "特禀质",
	"adl_walk": "ADL走路",
	"adl_eat": "ADL吃饭",
	"adl_dress": "ADL穿衣",
	"adl_toilet": "ADL如厕",
	"adl_bath": "ADL洗澡",
	"adl_total": "ADL总分",
	"iadl_shop": "IADL购物",
	"iadl_cook": "IADL做饭",
	"iadl_finance": "IADL理财",
	"iadl_transport": "IADL交通",
	"iadl_medicine": "IADL服药",
	"iadl_total": "IADL总分",
	"activity_total": "活动量表总分（ADL总分+IADL总分）",
	"hdl_c": "HDL-C（高密度脂蛋白胆固醇）",
	"ldl_c": "LDL-C（低密度脂蛋白胆固醇）",
	"tg": "TG（甘油三酯）",
	"tc": "TC（总胆固醇）",
	"glucose": "空腹血糖",
	"uric_acid": "血尿酸",
	"bmi": "BMI",
	"hyperlipidemia_label": "高血脂症标签",
	"hyperlipidemia_type": "血脂异常分型标签（确诊病例）",
	"age_group": "年龄组",
	"gender": "性别",
	"smoking": "吸烟史",
	"drinking": "饮酒史",
}

LIPID_LIMITS = {
	"tc": 6.2,
	"tg": 1.7,
	"ldl_c": 3.1,
	"hdl_c": 1.04,
}

TCM_COST = {1: 30.0, 2: 80.0, 3: 130.0}
TRAIN_UNIT_COST = {1: 3.0, 2: 5.0, 3: 8.0}
TCM_DROP_RATE = {1: 0.01, 2: 0.02, 3: 0.03}

P1_CANDIDATE_FEATURES = [
	"adl_total",
	"iadl_total",
	"activity_total",
	"tc",
	"tg",
	"ldl_c",
	"hdl_c",
	"glucose",
	"uric_acid",
	"bmi",
]

RISK_FEATURES = [
	"score_tanshi",
	"activity_total",
	"tc",
	"tg",
	"ldl_c",
	"hdl_c",
	"glucose",
	"uric_acid",
	"bmi",
	"age_group",
	"gender",
	"smoking",
	"drinking",
	"score_pinghe",
	"score_qixu",
	"score_yangxu",
	"score_yinxu",
	"score_shire",
	"score_xueyu",
	"score_qiyu",
	"score_tebing",
]


def show_name(key: str) -> str:
	return DISPLAY_NAMES.get(key, key)


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


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
	x0 = np.asarray(x, dtype=float)
	y0 = np.asarray(y, dtype=float)
	mask = np.isfinite(x0) & np.isfinite(y0)
	if mask.sum() < 3:
		return 0.0
	xs = x0[mask]
	ys = y0[mask]
	if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
		return 0.0
	return float(np.corrcoef(xs, ys)[0, 1])


def zscore(x: np.ndarray) -> np.ndarray:
	mu = np.nanmean(x)
	sd = np.nanstd(x)
	if sd < 1e-9:
		return np.zeros_like(x)
	return (x - mu) / sd


def sigmoid(z: np.ndarray) -> np.ndarray:
	z_clip = np.clip(z, -30.0, 30.0)
	return 1.0 / (1.0 + np.exp(-z_clip))


def fit_logistic_gd(
	X: np.ndarray,
	y: np.ndarray,
	lr: float = 0.03,
	max_iter: int = 6000,
	l2: float = 1e-3,
) -> Tuple[np.ndarray, float]:
	n, p = X.shape
	w = np.zeros(p, dtype=float)
	b = 0.0
	y = y.astype(float)

	for _ in range(max_iter):
		z = X @ w + b
		p_hat = sigmoid(z)
		err = p_hat - y

		grad_w = (X.T @ err) / n + l2 * w
		grad_b = float(np.mean(err))

		w -= lr * grad_w
		b -= lr * grad_b

	return w, b


def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y = y_true.astype(int)
	p = y_prob.astype(float)
	pos = p[y == 1]
	neg = p[y == 0]
	if len(pos) == 0 or len(neg) == 0:
		return 0.5
	count = 0.0
	total = float(len(pos) * len(neg))
	for vp in pos:
		count += float(np.sum(vp > neg)) + 0.5 * float(np.sum(vp == neg))
	return count / total


def lipid_abnormal_count(data: Dict[str, np.ndarray]) -> np.ndarray:
	tc_abn = (data["tc"] > LIPID_LIMITS["tc"]).astype(int)
	tg_abn = (data["tg"] > LIPID_LIMITS["tg"]).astype(int)
	ldl_abn = (data["ldl_c"] > LIPID_LIMITS["ldl_c"]).astype(int)
	hdl_low = (data["hdl_c"] < LIPID_LIMITS["hdl_c"]).astype(int)
	return tc_abn + tg_abn + ldl_abn + hdl_low


def problem1_key_indicator_analysis(data: Dict[str, np.ndarray]) -> Dict[str, object]:
	target_tanshi = data["score_tanshi"]
	target_risk = data["hyperlipidemia_label"]

	rows = []
	for f in P1_CANDIDATE_FEATURES:
		c1 = safe_corr(data[f], target_tanshi)
		c2 = safe_corr(data[f], target_risk)
		score = 0.6 * abs(c1) + 0.4 * abs(c2)
		rows.append((show_name(f), c1, c2, score))

	rows.sort(key=lambda x: x[3], reverse=True)

	# 九种体质贡献度：使用患病率提升倍数作为贡献度
	overall = float(np.mean(target_risk))
	constitution_contrib = []
	for k in range(1, 10):
		idx = data["constitution_label"] == k
		if np.sum(idx) == 0:
			rr = 1.0
			rate = overall
		else:
			rate = float(np.mean(target_risk[idx]))
			rr = rate / overall if overall > 1e-9 else 1.0
		constitution_contrib.append((k, rate, rr))

	constitution_contrib.sort(key=lambda x: x[2], reverse=True)

	return {
		"indicator_table": rows,
		"constitution_contrib": constitution_contrib,
	}


def build_risk_model(data: Dict[str, np.ndarray]) -> Dict[str, object]:
	y = data["hyperlipidemia_label"].astype(int)

	X_raw = np.column_stack([data[n] for n in RISK_FEATURES])
	X = np.column_stack([zscore(X_raw[:, i]) for i in range(X_raw.shape[1])])

	w, b = fit_logistic_gd(X, y)
	p = sigmoid(X @ w + b)
	auc = auc_score(y, p)

	lipid_cnt = lipid_abnormal_count(data)
	tanshi = data["score_tanshi"]
	activity = data["activity_total"]

	# 混合规则：医学阈值 + 概率评分
	high_rule = ((lipid_cnt >= 2) & (tanshi >= 60)) | ((tanshi >= 80) & (activity < 40)) | (p >= 0.70)
	mid_rule = (
		((lipid_cnt >= 1) & (tanshi >= 59))
		| ((tanshi >= 62) & (activity < 60))
		| ((p >= 0.40) & (p < 0.70))
	)

	risk_level = np.ones_like(y, dtype=int)
	risk_level[mid_rule] = 2
	risk_level[high_rule] = 3

	level_count = {
		"low": int(np.sum(risk_level == 1)),
		"medium": int(np.sum(risk_level == 2)),
		"high": int(np.sum(risk_level == 3)),
	}

	# 核心特征组合识别（高风险内统计支持度）
	high_idx = risk_level == 3
	if np.sum(high_idx) == 0:
		combo_stats = []
	else:
		conds = {
			"tanshi_high": tanshi >= 60,
			"activity_low": activity < 40,
			"lipid_multi_abn": lipid_cnt >= 2,
			"smoke_or_drink": (data["smoking"] == 1) | (data["drinking"] == 1),
		}
		combos = [
			("痰湿高+低活动+多血脂异常", ["tanshi_high", "activity_low", "lipid_multi_abn"]),
			("痰湿高+多血脂异常", ["tanshi_high", "lipid_multi_abn"]),
			("痰湿高+低活动", ["tanshi_high", "activity_low"]),
		]
		combo_stats = []
		n_high = int(np.sum(high_idx))
		for name, keys in combos:
			mask = np.ones_like(high_idx, dtype=bool)
			for k in keys:
				mask &= conds[k]
			support_in_high = float(np.sum(mask & high_idx)) / n_high
			confidence = float(np.sum(high_idx & mask)) / max(1, int(np.sum(mask)))
			combo_stats.append((name, support_in_high, confidence))
		combo_stats.sort(key=lambda x: x[1], reverse=True)

	# 权重绝对值用于特征解释
	coef_rank = sorted(
		[(show_name(RISK_FEATURES[i]), float(abs(w[i])), float(w[i])) for i in range(len(RISK_FEATURES))],
		key=lambda x: x[1],
		reverse=True,
	)

	return {
		"feature_names": RISK_FEATURES,
		"coef": w,
		"intercept": b,
		"prob": p,
		"auc": auc,
		"risk_level": risk_level,
		"level_count": level_count,
		"combo_stats": combo_stats,
		"coef_rank": coef_rank,
	}


@dataclass
class Action:
	tcm_level: int
	intensity: int
	freq_per_week: int


@dataclass
class PlanResult:
	best_cost: float
	final_score: int
	schedule: List[Action]


def allowed_intensity(age_group: int, activity_total: float) -> List[int]:
	if age_group == 5:
		by_age = {1}
	elif age_group in (3, 4):
		by_age = {1, 2}
	else:
		by_age = {1, 2, 3}

	if activity_total < 40:
		by_activity = {1}
	elif activity_total < 60:
		by_activity = {1, 2}
	else:
		by_activity = {1, 2, 3}

	return sorted(list(by_age & by_activity))


def tcm_level_by_score(score_tanshi: int) -> int:
	if score_tanshi <= 58:
		return 1
	if score_tanshi <= 61:
		return 2
	return 3


def one_month_transition(score: int, action: Action) -> Tuple[int, float]:
	tcm_cost = TCM_COST[action.tcm_level]
	train_unit_cost = TRAIN_UNIT_COST[action.intensity]

	month_cost = tcm_cost + train_unit_cost * action.freq_per_week * 4

	# 痰湿分月更新：训练效果+中医调理效果，线性近似
	train_drop_rate = max(0.0, 0.03 * (action.intensity - 1) + 0.01 * (action.freq_per_week - 5))
	tcm_drop_rate = TCM_DROP_RATE[action.tcm_level]
	total_drop_rate = min(0.35, train_drop_rate + tcm_drop_rate)

	new_score = int(round(max(0.0, score * (1.0 - total_drop_rate))))
	return new_score, month_cost


def optimize_patient_plan(
	init_score: int,
	age_group: int,
	activity_total: float,
	months: int = 6,
	budget_max: float = 2000.0,
) -> PlanResult:
	intensities = allowed_intensity(age_group, activity_total)
	freqs = list(range(1, 11))

	# DP: month -> score -> (cost, path)
	dp: Dict[int, Tuple[float, List[Action]]] = {int(init_score): (0.0, [])}

	for _ in range(months):
		ndp: Dict[int, Tuple[float, List[Action]]] = {}
		for score, (cost_now, path_now) in dp.items():
			level = tcm_level_by_score(score)
			for inten in intensities:
				for f in freqs:
					action = Action(level, inten, f)
					next_score, month_cost = one_month_transition(score, action)
					new_cost = cost_now + month_cost
					if new_cost > budget_max:
						continue
					old = ndp.get(next_score)
					if old is None or new_cost < old[0]:
						ndp[next_score] = (new_cost, path_now + [action])
		dp = ndp
		if not dp:
			break

	if not dp:
		# 若预算过紧导致不可行，则退化为低成本基础方案
		fallback_action = Action(tcm_level_by_score(init_score), intensities[0], 1)
		fallback = [fallback_action for _ in range(months)]
		cost = 0.0
		score = int(init_score)
		for a in fallback:
			score, c = one_month_transition(score, a)
			cost += c
		return PlanResult(best_cost=cost, final_score=score, schedule=fallback)

	best_score = min(dp.keys())
	best_cost, best_path = dp[best_score]
	return PlanResult(best_cost=best_cost, final_score=best_score, schedule=best_path)


def problem3_optimize(data: Dict[str, np.ndarray]) -> Dict[str, object]:
	idx = data["constitution_label"] == 5
	sample_ids = data["sample_id"][idx].astype(int)
	tanshi = data["score_tanshi"][idx].astype(int)
	age_group = data["age_group"][idx].astype(int)
	activity = data["activity_total"][idx]

	all_results = []
	for sid, s0, ag, act in zip(sample_ids, tanshi, age_group, activity):
		plan = optimize_patient_plan(int(s0), int(ag), float(act))
		all_results.append((int(sid), int(s0), int(ag), float(act), plan))

	# 样本 ID 1/2/3 单独输出
	sample_plan = {}
	for sid in (1, 2, 3):
		match = [r for r in all_results if r[0] == sid]
		if match:
			sample_plan[sid] = match[0]

	# 匹配规律：按年龄段+活动分层统计最常见强度与频次
	pattern_bucket: Dict[Tuple[int, str], List[PlanResult]] = {}
	for _, _, ag, act, plan in all_results:
		act_bin = "L" if act < 40 else ("M" if act < 60 else "H")
		pattern_bucket.setdefault((ag, act_bin), []).append(plan)

	pattern_summary = []
	for key, plans in pattern_bucket.items():
		all_actions = [a for p in plans for a in p.schedule]
		if not all_actions:
			continue
		avg_intensity = float(np.mean([a.intensity for a in all_actions]))
		avg_freq = float(np.mean([a.freq_per_week for a in all_actions]))
		avg_final = float(np.mean([p.final_score for p in plans]))
		avg_cost = float(np.mean([p.best_cost for p in plans]))
		pattern_summary.append((key[0], key[1], avg_intensity, avg_freq, avg_final, avg_cost, len(plans)))

	pattern_summary.sort(key=lambda x: (x[0], x[1]))

	return {
		"all_results": all_results,
		"sample_plan": sample_plan,
		"pattern_summary": pattern_summary,
	}


def explain_math_model() -> str:
	return "\n".join(
		[
			"数学模型说明（摘要）",
			"1) 指标筛选模型",
			f"- 目标1：表征{show_name('score_tanshi')}严重程度，目标变量为{show_name('score_tanshi')}。",
			f"- 目标2：预警{show_name('hyperlipidemia_label')}风险，目标变量为{show_name('hyperlipidemia_label')}。",
			f"- 对候选指标 x_j 计算双目标评分：S_j = 0.6*|corr(x_j, {show_name('score_tanshi')})| + 0.4*|corr(x_j, y)|。",
			"- 按 S_j 排序得到关键指标。",
			"2) 多维风险预警模型",
			"- 建立逻辑回归概率模型：p = sigma(w^T z + b)，z 为标准化特征。",
			"- 通过医学规则与概率阈值混合分层：",
			f"  高风险： ({show_name('hyperlipidemia_label')}异常项>=2 且 {show_name('score_tanshi')}>=60) 或 ({show_name('score_tanshi')}>=80 且 {show_name('activity_total')}<40) 或 p>=0.70。",
			"  中风险：满足若干中间规则或 0.40<=p<0.70。",
			"  低风险：其余样本。",
			"3) 干预优化模型（6个月）",
			f"- 状态变量：每月{show_name('score_tanshi')}。",
			"- 决策变量：中医调理级别 g_t、活动强度 i_t、每周频次 f_t。",
			f"- 状态转移：s_{{t+1}}=round(s_t*(1-r_t))，r_t = r_train(i_t,f_t)+r_tcm(g_t)，其中 s_t 对应{show_name('score_tanshi')}。",
			"- 成本函数：C = sum_t (C_tcm(g_t) + 4*f_t*C_train(i_t))，约束 C<=2000。",
			"- 在年龄/活动能力约束下用动态规划求解，使最终积分最小、同分时成本最小。",
		]
	)


def format_problem1_output(res: Dict[str, object]) -> str:
	lines = []
	lines.append("问题1结果")
	lines.append("关键指标排序（前8）:")
	for f, c1, c2, s in res["indicator_table"][:8]:
		lines.append(
			f"- {f:24s} 与{show_name('score_tanshi')}相关={c1:+.3f} 与{show_name('hyperlipidemia_label')}相关={c2:+.3f} 综合得分={s:.3f}"
		)

	lines.append("九种体质风险贡献（按相对风险倍数排序）:")
	for k, rate, rr in res["constitution_contrib"]:
		lines.append(f"- 体质{k}: 患病率={rate:.3f}, 相对风险RR={rr:.3f}")
	return "\n".join(lines)


def format_problem2_output(res: Dict[str, object]) -> str:
	lines = []
	lines.append("问题2结果")
	lines.append(f"模型AUC(训练集近似): {res['auc']:.4f}")
	lines.append(
		f"风险分层计数: low={res['level_count']['low']}, medium={res['level_count']['medium']}, high={res['level_count']['high']}"
	)
	lines.append("高风险核心组合统计（support@high / confidence）:")
	for name, sup, conf in res["combo_stats"]:
		lines.append(f"- {name}: {sup:.3f} / {conf:.3f}")
	lines.append("模型权重绝对值Top10（字段名使用样例原始字段）:")
	for name, absw, w in res["coef_rank"][:10]:
		lines.append(f"- {name:14s} |w|={absw:.4f}, w={w:+.4f}")
	return "\n".join(lines)


def format_schedule(plan: PlanResult) -> str:
	parts = []
	for i, a in enumerate(plan.schedule, 1):
		parts.append(
			f"第{i}月:中医调理{a.tcm_level}级,活动干预强度{a.intensity}级,每周{a.freq_per_week}次"
		)
	return "；".join(parts)


def format_problem3_output(res: Dict[str, object]) -> str:
	lines = []
	lines.append("问题3结果")
	lines.append("样本ID=1/2/3最优方案:")
	for sid in (1, 2, 3):
		if sid not in res["sample_plan"]:
			lines.append(f"- {show_name('sample_id')}={sid}: 样本不存在")
			continue
		sid0, s0, ag, act, plan = res["sample_plan"][sid]
		lines.append(
			f"- {show_name('sample_id')}={sid0}, 初始{show_name('score_tanshi')}={s0}, {show_name('age_group')}={ag}, {show_name('activity_total')}={act:.1f}, 6月后{show_name('score_tanshi')}={plan.final_score}, 总成本={plan.best_cost:.1f}"
		)
		lines.append(f"  方案: {format_schedule(plan)}")

	lines.append("患者特征-最优方案匹配规律(分层均值):")
	lines.append(f"- 字段: {show_name('age_group')}, {show_name('activity_total')}分层(L<40/M40-59/H>=60), 平均强度, 平均频次, 平均终点{show_name('score_tanshi')}, 平均成本, 样本数")
	for ag, act_bin, ai, af, fs, ac, n in res["pattern_summary"]:
		lines.append(
			f"- {show_name('age_group')}={ag}, 活动层={act_bin}, 平均活动强度={ai:.2f}, 平均每周频次={af:.2f}, 平均终点{show_name('score_tanshi')}={fs:.2f}, 平均成本={ac:.2f}, 样本数={n}"
		)
	return "\n".join(lines)


def autosize_columns(ws) -> None:
	for column_cells in ws.columns:
		max_length = 0
		column_letter = column_cells[0].column_letter
		for cell in column_cells:
			value = "" if cell.value is None else str(cell.value)
			if len(value) > max_length:
				max_length = len(value)
		ws.column_dimensions[column_letter].width = min(max_length + 2, 60)


def write_text_report(path: Path, report: str) -> None:
	path.write_text(report, encoding="utf-8")


def write_xlsx_report(path: Path, report: str, p1: Dict[str, object], p2: Dict[str, object], p3: Dict[str, object]) -> None:
	wb = Workbook()
	default_ws = wb.active
	default_ws.title = "总览"
	default_ws.sheet_view.showGridLines = False

	for line in report.splitlines():
		default_ws.append([line])
	default_ws.column_dimensions["A"].width = 120

	ws1 = wb.create_sheet("问题1")
	ws1.append(["关键指标排序"])
	ws1.append(["指标", "与痰湿质相关", "与高血脂症相关", "综合得分"])
	for f, c1, c2, s in p1["indicator_table"]:
		ws1.append([f, float(c1), float(c2), float(s)])
	ws1.append([])
	ws1.append(["九种体质风险贡献"])
	ws1.append(["体质编号", "患病率", "相对风险RR"])
	for k, rate, rr in p1["constitution_contrib"]:
		ws1.append([int(k), float(rate), float(rr)])
	autosize_columns(ws1)

	ws2 = wb.create_sheet("问题2")
	ws2.append(["模型指标"])
	ws2.append(["指标", "值"])
	ws2.append(["AUC(训练集近似)", float(p2["auc"])])
	ws2.append(["low", int(p2["level_count"]["low"])])
	ws2.append(["medium", int(p2["level_count"]["medium"])])
	ws2.append(["high", int(p2["level_count"]["high"])])
	ws2.append([])
	ws2.append(["高风险核心组合统计"])
	ws2.append(["组合", "support@high", "confidence"])
	for name, sup, conf in p2["combo_stats"]:
		ws2.append([name, float(sup), float(conf)])
	ws2.append([])
	ws2.append(["模型权重绝对值Top10"])
	ws2.append(["字段", "|w|", "w"])
	for name, absw, w in p2["coef_rank"][:10]:
		ws2.append([name, float(absw), float(w)])
	autosize_columns(ws2)

	ws3 = wb.create_sheet("问题3")
	ws3.append(["样本ID最优方案"])
	ws3.append(["样本ID", "初始痰湿质", "年龄组", "活动量表总分", "6月后痰湿质", "总成本", "第1-6月方案"])
	for sid in (1, 2, 3):
		if sid not in p3["sample_plan"]:
			ws3.append([sid, "不存在", "", "", "", "", ""])
			continue
		sid0, s0, ag, act, plan = p3["sample_plan"][sid]
		ws3.append([
			int(sid0),
			int(s0),
			int(ag),
			float(act),
			int(plan.final_score),
			float(plan.best_cost),
			format_schedule(plan),
		])
	ws3.append([])
	ws3.append(["患者特征-最优方案匹配规律"])
	ws3.append(["年龄组", "活动层", "平均强度", "平均频次", "平均终点痰湿质", "平均成本", "样本数"])
	for ag, act_bin, ai, af, fs, ac, n in p3["pattern_summary"]:
		ws3.append([int(ag), act_bin, float(ai), float(af), float(fs), float(ac), int(n)])
	autosize_columns(ws3)

	wb.save(path)


def run(data_path: Path, out_path: Path) -> None:
	data = load_xlsx(data_path)

	p1 = problem1_key_indicator_analysis(data)
	p2 = build_risk_model(data)
	p3 = problem3_optimize(data)

	text_blocks = [
		explain_math_model(),
		"",
		format_problem1_output(p1),
		"",
		format_problem2_output(p2),
		"",
		format_problem3_output(p3),
	]
	report = "\n".join(text_blocks)
	if out_path.suffix.lower() == ".xlsx":
		write_xlsx_report(out_path, report, p1, p2, p3)
	else:
		write_text_report(out_path, report)
	print(report)
	print(f"\n分析报告已输出: {out_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="MathorCup C题：高血脂风险预警与干预优化")
	parser.add_argument(
		"--data",
		type=str,
		default=r"e:\Code\C题\附件1：样例数据.xlsx",
		help="样例数据Excel路径",
	)
	parser.add_argument(
		"--out",
		type=str,
		default=r"e:\Code\C题\indx\result_report.xlsx",
		help="输出报告路径（.txt 保持文本，.xlsx 输出表格）",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(Path(args.data), Path(args.out))
