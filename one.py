#%%
"""
원예장비 제조업체 총괄생산계획 (Aggregate Production Planning) 웹앱
─────────────────────────────────────────────────────────────────────
강의록 「스마트제조 06 — 총괄생산계획」 모델 (Pyomo LP/IP) 을 그대로 구현하고,
표준 휴리스틱 전략(Level / Chase / Mixed)과 비교하는 의사결정 지원 도구.

저자: 산업공학 프로젝트
"""

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="원예장비 총괄생산계획",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 강의록 기준 기본값
# =============================================================================
DEFAULT_DEMAND = [1600, 3000, 3200, 3800, 2200, 2200]   # 강의록 1월~6월

DEFAULT_PARAMS = dict(
    initial_workforce=80,           # W_0
    initial_inventory=1000,         # I_0
    final_inventory_min=500,        # I_N >= 500
    workdays=20,                    # 작업일수 (일/월)
    work_hours_per_day=8,           # 작업시간 (시간/일)
    hours_per_unit=4.0,             # 작업표준시간 (시간/개)
    max_ot_hours_per_worker=10,     # 초과시간 제한 (시간/인·월)
    regular_wage=4,                 # 정규임금 (천원/시)
    overtime_wage=6,                # 초과근무 임금 (천원/시)
    hiring_cost=300,                # 고용비 (천원/인)
    firing_cost=500,                # 해고비 (천원/인)
    holding_cost=2,                 # 재고유지비 (천원/개·월)
    backorder_cost=5,               # 부재고비 (천원/개·월)
    material_cost=10,               # 재료비 (천원/개)
    subcontract_cost=30,            # 외주 추가비용 (천원/개)
    max_ot_ratio=0.20,              # Mixed 전략용
)

COST_COLS = ["정규임금비", "잔업임금비", "고용비", "해고비",
             "재고비", "결품비", "외주비", "자재비"]


def get_months(n: int) -> List[str]:
    return [f"{i}월" for i in range(1, n + 1)]


def units_per_worker(p: Dict) -> float:
    """1명의 작업자가 한 달 정규시간에 생산 가능한 단위 수
    = 작업일수 × 작업시간/일 ÷ 작업표준시간
    강의록 기본값: 20 × 8 ÷ 4 = 40 대/월"""
    return p["workdays"] * p["work_hours_per_day"] / p["hours_per_unit"]


def labor_cost_per_month(p: Dict) -> float:
    """1명의 작업자 정규임금비 (천원/월)
    강의록 기본값: 20 × 8 × 4 = 640 천원/월"""
    return p["workdays"] * p["work_hours_per_day"] * p["regular_wage"]


# =============================================================================
# Pyomo / PuLP 최적화 모델 (강의록 모델 그대로)
# =============================================================================
def solve_optimization(demand: List[float], params: Dict,
                       integer: bool = False, name: str = None):
    """강의록의 Pyomo 최적화 모델을 PuLP(동등한 LP/MIP 모델러)로 구현.

    결정변수 (모두 ≥ 0):
        W_t: t월의 종업원 수
        H_t: t월초 고용
        L_t: t월초 해고
        P_t: t월 생산량 (정규+잔업 합계)
        I_t: t월 말 재고
        S_t: t월 말 부족재고(backorder)
        C_t: t월 외주 수량
        O_t: t월 잔업시간 합계 (시간)

    목적함수 (강의록 식):
        Z = Σ ( 640·W_t + 6·O_t + 300·H_t + 500·L_t
                + 2·I_t + 5·S_t + 10·P_t + 30·C_t )

    제약:
        W_t = W_{t-1} + H_t - L_t                  (노동력)
        P_t ≤ 40·W_t + 0.25·O_t                    (생산능력)
        I_t = I_{t-1} + P_t + C_t - D_t - S_{t-1} + S_t  (재고균형)
        O_t ≤ 10·W_t                               (잔업한도)
        W_0 = 80, I_0 = 1000, S_0 = 0
        I_N ≥ 500, S_N = 0
    """
    try:
        import pulp
    except ImportError:
        st.error("PuLP 패키지가 필요합니다. `pip install pulp` 실행 후 재시도하세요.")
        return None

    N = len(demand)
    cat = "Integer" if integer else "Continuous"
    if name is None:
        name = f"Pyomo {'IP' if integer else 'LP'}"

    prob = pulp.LpProblem("APP", pulp.LpMinimize)

    T_all = range(0, N + 1)        # 0=초기, 1..N=계획
    T = range(1, N + 1)            # 계획 기간

    W = {t: pulp.LpVariable(f"W_{t}", lowBound=0, cat=cat) for t in T_all}
    H = {t: pulp.LpVariable(f"H_{t}", lowBound=0, cat=cat) for t in T_all}
    L = {t: pulp.LpVariable(f"L_{t}", lowBound=0, cat=cat) for t in T_all}
    P = {t: pulp.LpVariable(f"P_{t}", lowBound=0, cat=cat) for t in T_all}
    I = {t: pulp.LpVariable(f"I_{t}", lowBound=0, cat=cat) for t in T_all}
    S = {t: pulp.LpVariable(f"S_{t}", lowBound=0, cat=cat) for t in T_all}
    C = {t: pulp.LpVariable(f"C_{t}", lowBound=0, cat=cat) for t in T_all}
    O = {t: pulp.LpVariable(f"O_{t}", lowBound=0, cat="Continuous") for t in T_all}

    upw = units_per_worker(params)
    lcp = labor_cost_per_month(params)

    # ─── 목적함수 ──────────────────────────────────────────────
    prob += pulp.lpSum(
        lcp * W[t]
        + params["overtime_wage"]    * O[t]
        + params["hiring_cost"]      * H[t]
        + params["firing_cost"]      * L[t]
        + params["holding_cost"]     * I[t]
        + params["backorder_cost"]   * S[t]
        + params["material_cost"]    * P[t]
        + params["subcontract_cost"] * C[t]
        for t in T
    )

    # ─── 초기/최종 조건 ────────────────────────────────────────
    prob += W[0] == params["initial_workforce"]
    prob += I[0] == params["initial_inventory"]
    prob += S[0] == 0
    for v in (H, L, P, C, O):
        prob += v[0] == 0
    prob += I[N] >= params["final_inventory_min"]
    prob += S[N] == 0

    # ─── 기간별 제약 ──────────────────────────────────────────
    for t in T:
        prob += W[t] == W[t-1] + H[t] - L[t]
        prob += P[t] <= upw * W[t] + O[t] / params["hours_per_unit"]
        prob += I[t] - S[t] == I[t-1] - S[t-1] + P[t] + C[t] - demand[t-1]
        prob += O[t] <= params["max_ot_hours_per_worker"] * W[t]

    # ─── 풀이 ─────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    status = prob.solve(solver)

    if pulp.LpStatus[status] not in ("Optimal",):
        st.warning(f"최적화 풀이 실패: {pulp.LpStatus[status]}")
        return None

    # ─── 결과 → DataFrame ────────────────────────────────────
    months = get_months(N)
    rnd = (lambda x: int(round(x))) if integer else (lambda x: round(float(x), 2))

    rows = []
    for t in T:
        w_v = W[t].value()
        ot_h = O[t].value()
        p_total = P[t].value()
        # 정규/잔업 분리 추정
        regular_cap = upw * w_v
        if p_total <= regular_cap + 1e-6:
            reg_used = p_total
            ot_used = 0.0
        else:
            reg_used = regular_cap
            ot_used = p_total - regular_cap

        rows.append({
            "월": months[t-1],
            "수요": int(demand[t-1]),
            "인력": rnd(w_v),
            "고용": rnd(H[t].value()),
            "해고": rnd(L[t].value()),
            "정규생산": rnd(reg_used),
            "잔업시간": round(ot_h, 2),
            "잔업생산": rnd(ot_used),
            "외주": rnd(C[t].value()),
            "총공급": rnd(p_total + C[t].value()),
            "기말재고": rnd(I[t].value()),
            "결품": rnd(S[t].value()),
        })

    df = pd.DataFrame(rows)

    df["정규임금비"] = df["인력"]    * lcp
    df["잔업임금비"] = df["잔업시간"] * params["overtime_wage"]
    df["고용비"]     = df["고용"]    * params["hiring_cost"]
    df["해고비"]     = df["해고"]    * params["firing_cost"]
    df["재고비"]     = df["기말재고"] * params["holding_cost"]
    df["결품비"]     = df["결품"]    * params["backorder_cost"]
    df["외주비"]     = df["외주"]    * params["subcontract_cost"]
    df["자재비"]     = (df["정규생산"] + df["잔업생산"]) * params["material_cost"]
    df["월총비용"]   = df[COST_COLS].sum(axis=1)

    df.attrs["name"] = name
    df.attrs["objective_value"] = float(pulp.value(prob.objective))
    df.attrs["solver_status"] = pulp.LpStatus[status]
    df.attrs["params"] = params
    return df


# =============================================================================
# 휴리스틱 전략 (강의 휴리스틱 baseline)
# =============================================================================
def compute_plan(demand, params, workforce, overtime_units, subcon, name="Plan"):
    """workforce/overtime(단위)/subcon(단위) 결정변수가 주어진 경우 비용 계산."""
    upw = units_per_worker(params)
    lcp = labor_cost_per_month(params)
    months = get_months(len(demand))
    rows = []
    pw = params["initial_workforce"]
    pi = params["initial_inventory"]

    for t, d in enumerate(demand):
        w = int(workforce[t])
        h = max(0, w - pw)
        f = max(0, pw - w)
        rp = w * upw
        ot_units = max(0, int(overtime_units[t]))
        ot_hours = ot_units * params["hours_per_unit"]
        sc = max(0, int(subcon[t]))
        net = pi + rp + ot_units + sc - d
        inv = max(net, 0)
        bo = max(-net, 0)

        rows.append({
            "월": months[t],
            "수요": int(d),
            "인력": w, "고용": h, "해고": f,
            "정규생산": int(round(rp)),
            "잔업시간": round(ot_hours, 1),
            "잔업생산": ot_units,
            "외주": sc,
            "총공급": int(round(rp)) + ot_units + sc,
            "기말재고": int(round(inv)),
            "결품": int(round(bo)),
        })
        pw, pi = w, net

    df = pd.DataFrame(rows)
    df["정규임금비"] = df["인력"]    * lcp
    df["잔업임금비"] = df["잔업시간"] * params["overtime_wage"]
    df["고용비"]     = df["고용"]    * params["hiring_cost"]
    df["해고비"]     = df["해고"]    * params["firing_cost"]
    df["재고비"]     = df["기말재고"] * params["holding_cost"]
    df["결품비"]     = df["결품"]    * params["backorder_cost"]
    df["외주비"]     = df["외주"]    * params["subcontract_cost"]
    df["자재비"]     = (df["정규생산"] + df["잔업생산"]) * params["material_cost"]
    df["월총비용"]   = df[COST_COLS].sum(axis=1)
    df.attrs["name"] = name
    return df


def plan_level(demand, params):
    """Level: 평균 수요를 충족할 수 있는 인력으로 일정하게 유지."""
    upw = units_per_worker(params)
    n = int(np.ceil(np.mean(demand) / upw))
    return compute_plan(demand, params, [n]*len(demand),
                        [0]*len(demand), [0]*len(demand), "Level")


def plan_chase(demand, params):
    """Chase: 매월 수요에 맞춰 인력 조정."""
    upw = units_per_worker(params)
    workforce = [max(1, int(np.ceil(d / upw))) for d in demand]
    return compute_plan(demand, params, workforce,
                        [0]*len(demand), [0]*len(demand), "Chase")


def plan_mixed(demand, params, base_workers=None,
               overtime_ratio=None, allow_subcontract=True):
    """Mixed: 기준 인력 고정 + 잔업·외주로 변동 흡수."""
    upw = units_per_worker(params)
    if base_workers is None:
        base_workers = int(np.ceil((np.min(demand) + np.mean(demand)) / 2 / upw))
    if overtime_ratio is None:
        overtime_ratio = params["max_ot_ratio"]
    cap = base_workers * upw
    max_ot = int(cap * overtime_ratio)

    ot_list, sc_list = [], []
    pi = params["initial_inventory"]
    for d in demand:
        need = d - cap - max(pi, 0) * 0.5
        if need > 0:
            o = min(need, max_ot)
            s = max(0, need - o) if allow_subcontract else 0
        else:
            o, s = 0, 0
        ot_list.append(int(round(o)))
        sc_list.append(int(round(s)))
        pi = pi + cap + o + s - d
    return compute_plan(demand, params, [base_workers]*len(demand),
                        ot_list, sc_list, "Mixed")


def plan_custom(demand, params, workforce, overtime_units, subcon):
    return compute_plan(demand, params, workforce, overtime_units, subcon, "Custom")


# =============================================================================
# 시각화 함수
# =============================================================================
def fig_demand_supply(df: pd.DataFrame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=df["월"], y=df["수요"], name="수요",
                marker_color="rgba(99,110,250,0.55)")
    fig.add_scatter(x=df["월"], y=df["정규생산"], mode="lines+markers",
                    name="정규생산", line=dict(color="#2ca02c", width=3))
    fig.add_scatter(x=df["월"], y=df["총공급"], mode="lines+markers",
                    name="총공급", line=dict(color="#ff7f0e", width=3, dash="dash"))
    fig.add_scatter(x=df["월"], y=df["기말재고"], mode="lines+markers",
                    name="기말재고", line=dict(color="#1f77b4", width=2),
                    secondary_y=True)
    if df["결품"].sum() > 0:
        fig.add_bar(x=df["월"], y=-df["결품"], name="결품",
                    marker_color="rgba(214,39,40,0.6)", secondary_y=True)
    fig.update_layout(
        title=f"{df.attrs['name']} — 수요·공급·재고 추이",
        height=420, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="수량 (대)", secondary_y=False)
    fig.update_yaxes(title_text="재고/결품 (대)", secondary_y=True)
    return fig


def fig_workforce(df: pd.DataFrame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=df["월"], y=df["인력"], mode="lines+markers",
                    name="인력 수", line=dict(color="#9467bd", width=3))
    fig.add_bar(x=df["월"], y=df["고용"], name="고용",
                marker_color="#2ca02c", secondary_y=True, opacity=0.7)
    fig.add_bar(x=df["월"], y=-df["해고"], name="해고",
                marker_color="#d62728", secondary_y=True, opacity=0.7)
    fig.update_layout(
        title=f"{df.attrs['name']} — 인력 변동",
        height=380, hovermode="x unified", barmode="relative",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="인력 (명)", secondary_y=False)
    fig.update_yaxes(title_text="고용/해고 (명)", secondary_y=True)
    return fig


def fig_cost_breakdown(df: pd.DataFrame):
    fig = go.Figure()
    for c in COST_COLS:
        fig.add_bar(x=df["월"], y=df[c] / 1_000, name=c)   # 천원→백만원? aim 천원
    fig.update_layout(
        title=f"{df.attrs['name']} — 월별 비용 분해 (단위: 백만원)",
        barmode="stack", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="비용 (백만원)")
    return fig


def fig_strategy_compare_cost(plans: Dict[str, pd.DataFrame]):
    rows = []
    totals = {}
    for name, df in plans.items():
        total = df["월총비용"].sum() / 1_000
        totals[name] = total
        for c in COST_COLS:
            rows.append(dict(전략=name, 비용항목=c, 금액=df[c].sum() / 1_000))
    cdf = pd.DataFrame(rows)
    fig = px.bar(cdf, x="전략", y="금액", color="비용항목",
                 title="전략별 총비용 분해 (단위: 백만원)",
                 text_auto=".0f")
    # 각 막대 위에 총합 라벨 표시
    for name, total in totals.items():
        fig.add_annotation(
            x=name, y=total + max(totals.values()) * 0.04,
            text=f"<b>총 {total:,.1f}</b>",
            showarrow=False,
            font=dict(size=14, color="#FFD700"),
        )
    fig.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        xaxis=dict(
            title=dict(text="전략", font=dict(size=14)),
            tickfont=dict(size=14, color="#FFFFFF"),
        ),
        margin=dict(t=60, b=80),
    )
    fig.update_yaxes(title_text="총비용 (백만원)",
                     range=[0, max(totals.values()) * 1.15])
    return fig


def fig_strategy_kpi_radar(plans: Dict[str, pd.DataFrame]):
    metrics = {
        "총비용":      lambda df: df["월총비용"].sum(),
        "인력 변동":   lambda df: df["고용"].sum() + df["해고"].sum(),
        "평균 재고":   lambda df: df["기말재고"].mean(),
        "총 결품":     lambda df: df["결품"].sum() + 1,
        "외주 의존":   lambda df: df["외주"].sum() + 1,
    }
    raw = {m: [f(df) for df in plans.values()] for m, f in metrics.items()}
    norm = {m: [v / (max(vs) if max(vs) > 0 else 1) for v in vs] for m, vs in raw.items()}

    fig = go.Figure()
    for i, name in enumerate(plans.keys()):
        fig.add_trace(go.Scatterpolar(
            r=[norm[m][i] for m in metrics] + [norm[list(metrics)[0]][i]],
            theta=list(metrics.keys()) + [list(metrics.keys())[0]],
            fill='toself', name=name, opacity=0.45,
        ))
    fig.update_layout(
        title="전략별 KPI 비교 (값이 작을수록 우수, 0~1 정규화)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=470,
    )
    return fig


def fig_capacity_utilization(df: pd.DataFrame, params: Dict):
    upw = units_per_worker(params)
    cap = df["인력"] * upw
    util = (df["수요"] / cap.replace(0, np.nan)) * 100
    fig = go.Figure()
    fig.add_bar(x=df["월"], y=util,
                marker_color=np.where(util > 100, "#d62728", "#2ca02c"),
                name="가동률 (%)", text=[f"{u:.0f}%" for u in util],
                textposition="outside")
    fig.add_hline(y=100, line_dash="dash", line_color="gray",
                  annotation_text="100% 정규 capacity")
    fig.update_layout(
        title=f"{df.attrs['name']} — 정규생산 가동률",
        height=350, yaxis_title="가동률 (%)"
    )
    return fig


# =============================================================================
# 사이드바 (입력)
# =============================================================================
st.sidebar.title("⚙️ 입력 파라미터")
st.sidebar.caption("값을 바꾸면 모든 탭이 자동으로 다시 계산됩니다.")

n_periods = st.sidebar.slider("📅 계획 기간 (개월)", 3, 12,
                              value=len(DEFAULT_DEMAND))

# ── 수요 입력값 session_state 초기화 (위젯 생성 전에 default 세팅) ──
for i in range(n_periods):
    key = f"d_{i}_{n_periods}"
    if key not in st.session_state:
        st.session_state[key] = DEFAULT_DEMAND[i] if i < len(DEFAULT_DEMAND) else 2200


def _reset_demand():
    """기본값 복원 콜백 — 위젯 렌더링 전에 호출되어 안전하게 state 변경."""
    for i in range(n_periods):
        st.session_state[f"d_{i}_{n_periods}"] = (
            DEFAULT_DEMAND[i] if i < len(DEFAULT_DEMAND) else 2200
        )


with st.sidebar.expander("📈 월별 수요 (대)", expanded=True):
    demand_input = []
    cols = st.columns(2)
    for i in range(n_periods):
        with cols[i % 2]:
            demand_input.append(st.number_input(
                f"{i+1}월", min_value=0, max_value=20_000,
                step=100, key=f"d_{i}_{n_periods}"
            ))
    st.button("🔄 강의록 기본값 복원", use_container_width=True,
              on_click=_reset_demand)

with st.sidebar.expander("👷 초기/최종 조건"):
    initial_workforce = st.number_input("초기 인력 W₀ (명)", 1, 500,
                                        DEFAULT_PARAMS["initial_workforce"])
    initial_inventory = st.number_input("초기 재고 I₀ (대)", 0, 50_000,
                                        DEFAULT_PARAMS["initial_inventory"])
    final_inventory_min = st.number_input("최종 재고 하한 Iₙ (대)", 0, 50_000,
                                          DEFAULT_PARAMS["final_inventory_min"])

with st.sidebar.expander("🏭 작업/생산 파라미터"):
    workdays = st.number_input("작업일수 (일/월)", 1, 31,
                               DEFAULT_PARAMS["workdays"])
    work_hours_per_day = st.number_input("작업시간 (시간/일)", 1, 12,
                                         DEFAULT_PARAMS["work_hours_per_day"])
    hours_per_unit = st.number_input("작업표준시간 (시간/개)", 0.5, 50.0,
                                     DEFAULT_PARAMS["hours_per_unit"], step=0.5)
    max_ot_hours_per_worker = st.number_input("초과시간 한도 (시간/인·월)", 0, 50,
                                              DEFAULT_PARAMS["max_ot_hours_per_worker"])
    max_ot_ratio = st.slider("Mixed 휴리스틱: 최대 잔업 비율", 0.0, 0.5,
                             DEFAULT_PARAMS["max_ot_ratio"], step=0.05)

with st.sidebar.expander("💰 비용 파라미터 (천원 단위)"):
    regular_wage     = st.number_input("정규임금 (천원/시)",      0, 100,    DEFAULT_PARAMS["regular_wage"])
    overtime_wage    = st.number_input("초과근무 임금 (천원/시)",  0, 200,    DEFAULT_PARAMS["overtime_wage"])
    hiring_cost      = st.number_input("고용비 (천원/인)",        0, 10_000, DEFAULT_PARAMS["hiring_cost"], step=10)
    firing_cost      = st.number_input("해고비 (천원/인)",        0, 10_000, DEFAULT_PARAMS["firing_cost"], step=10)
    holding_cost     = st.number_input("재고유지비 (천원/대·월)", 0, 1_000,  DEFAULT_PARAMS["holding_cost"])
    backorder_cost   = st.number_input("부재고비 (천원/대·월)",   0, 1_000,  DEFAULT_PARAMS["backorder_cost"])
    material_cost    = st.number_input("재료비 (천원/대)",         0, 10_000, DEFAULT_PARAMS["material_cost"])
    subcontract_cost = st.number_input("외주 추가비용 (천원/대)", 0, 10_000, DEFAULT_PARAMS["subcontract_cost"])

with st.sidebar.expander("🏆 Pyomo 최적화"):
    use_integer = st.checkbox("정수계획법(IP) 사용 — 모든 변수 정수", value=False,
                              help="체크 해제 시 LP (연속 변수). 강의록과 동일.")

# 파라미터 dict
params = dict(
    initial_workforce=initial_workforce,
    initial_inventory=initial_inventory,
    final_inventory_min=final_inventory_min,
    workdays=workdays,
    work_hours_per_day=work_hours_per_day,
    hours_per_unit=hours_per_unit,
    max_ot_hours_per_worker=max_ot_hours_per_worker,
    max_ot_ratio=max_ot_ratio,
    regular_wage=regular_wage,
    overtime_wage=overtime_wage,
    hiring_cost=hiring_cost,
    firing_cost=firing_cost,
    holding_cost=holding_cost,
    backorder_cost=backorder_cost,
    material_cost=material_cost,
    subcontract_cost=subcontract_cost,
)
demand = list(demand_input)


# =============================================================================
# 전략 계산 (캐싱)
# =============================================================================
@st.cache_data(show_spinner=False)
def calc_heuristics(demand_t: tuple, p: tuple) -> Dict[str, pd.DataFrame]:
    pdict = dict(p)
    d = list(demand_t)
    return {
        "Level":  plan_level(d, pdict),
        "Chase":  plan_chase(d, pdict),
        "Mixed":  plan_mixed(d, pdict),
    }


@st.cache_data(show_spinner="Pyomo 최적화 풀이 중…")
def calc_optimum(demand_t: tuple, p: tuple, integer: bool):
    pdict = dict(p)
    d = list(demand_t)
    return solve_optimization(d, pdict, integer=integer,
                              name=f"Pyomo {'IP' if integer else 'LP'}")


params_t = tuple(sorted(params.items()))
heuristics = calc_heuristics(tuple(demand), params_t)
optimum_df = calc_optimum(tuple(demand), params_t, use_integer)

all_plans = {}
if optimum_df is not None:
    all_plans[optimum_df.attrs["name"]] = optimum_df
all_plans.update(heuristics)


# =============================================================================
# 본문 헤더
# =============================================================================
st.title("🌿 원예장비 제조업체 — 총괄생산계획(APP)")
st.markdown(
    f"강의록 「스마트제조 06 — 총괄생산계획」 의 **Pyomo LP/IP 모델**을 그대로 구현하고, "
    f"**Level / Chase / Mixed** 표준 휴리스틱과 비교합니다. "
    f"왼쪽 사이드바에서 수요·파라미터를 변경하면 모든 결과가 즉시 갱신됩니다."
)

# 핵심 KPI (4개 전략)
st.subheader("🏆 전략별 핵심 KPI")
kpi_cols = st.columns(len(all_plans))
for i, (name, df) in enumerate(all_plans.items()):
    with kpi_cols[i]:
        total_mil = df["월총비용"].sum() / 1_000   # 천원→백만원
        chg = df["고용"].sum() + df["해고"].sum()
        bo = df["결품"].sum()
        prefix = "🏆 " if name.startswith("Pyomo") else "📌 "
        st.metric(
            label=f"{prefix}{name}",
            value=f"{total_mil:,.1f} 백만원",
            delta=f"인력변동 {chg:.0f}명 / 결품 {bo:.0f}대",
            delta_color="off"
        )


# =============================================================================
# 탭
# =============================================================================
tabs = st.tabs([
    "🏆 Pyomo 최적화",
    "📊 종합 대시보드",
    "📋 전략별 상세",
    "🎛️ Custom 시나리오",
    "📥 데이터 / 보고서",
    "📚 모델 설명",
])
tab_opt, tab_dash, tab_strategy, tab_custom, tab_data, tab_help = tabs

# -----------------------------------------------------------------------------
with tab_opt:
    st.markdown("### 강의록 모델 (Pyomo LP/IP) 최적해")
    if optimum_df is None:
        st.error("최적해를 구하지 못했습니다. 파라미터·수요 값을 확인하세요.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최적 총비용", f"{optimum_df.attrs['objective_value']/1_000:,.1f} 백만원")
        c2.metric("계획 기간", f"{len(demand)} 개월")
        c3.metric("변수 유형", "정수(IP)" if use_integer else "연속(LP)")
        c4.metric("Solver 상태", optimum_df.attrs.get("solver_status", "—"))

        st.info(
            f"**목적함수**: Z = Σ ({labor_cost_per_month(params):.0f}·W_t "
            f"+ {params['overtime_wage']}·O_t + {params['hiring_cost']}·H_t "
            f"+ {params['firing_cost']}·L_t + {params['holding_cost']}·I_t "
            f"+ {params['backorder_cost']}·S_t + {params['material_cost']}·P_t "
            f"+ {params['subcontract_cost']}·C_t)  [단위: 천원]"
        )

        st.plotly_chart(fig_demand_supply(optimum_df),
                        use_container_width=True, key="opt_ds")
        cA, cB = st.columns(2)
        with cA:
            st.plotly_chart(fig_workforce(optimum_df),
                            use_container_width=True, key="opt_wf")
        with cB:
            st.plotly_chart(fig_capacity_utilization(optimum_df, params),
                            use_container_width=True, key="opt_cap")
        st.plotly_chart(fig_cost_breakdown(optimum_df),
                        use_container_width=True, key="opt_cost")

        st.markdown("##### 최적 의사결정 (강의록 표 형식)")
        show_cols = ["월","수요","인력","고용","해고","정규생산","잔업시간",
                     "잔업생산","외주","총공급","기말재고","결품"]
        st.dataframe(optimum_df[show_cols], use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
with tab_dash:
    st.markdown("### 1. 전략별 총비용 분해")
    st.plotly_chart(fig_strategy_compare_cost(all_plans),
                    use_container_width=True, key="dash_cost")

    st.markdown("### 2. KPI 레이더 (값이 작을수록 우수)")
    st.plotly_chart(fig_strategy_kpi_radar(all_plans),
                    use_container_width=True, key="dash_radar")

    st.markdown("### 3. 전략별 수요·공급 비교")
    cols = st.columns(min(len(all_plans), 4))
    for i, (name, df) in enumerate(all_plans.items()):
        with cols[i % len(cols)]:
            st.plotly_chart(fig_demand_supply(df),
                            use_container_width=True, key=f"dash_ds_{i}")

    st.markdown("### 4. 인사이트")
    if optimum_df is not None:
        opt_total = optimum_df["월총비용"].sum() / 1_000
        worst_name, worst_df = max(heuristics.items(),
                                   key=lambda x: x[1]["월총비용"].sum())
        worst_total = worst_df["월총비용"].sum() / 1_000
        savings = worst_total - opt_total
        st.success(
            f"💡 **Pyomo 최적해는 {opt_total:,.1f} 백만원**이며, "
            f"가장 비싼 휴리스틱({worst_name}: {worst_total:,.1f} 백만원) 대비 "
            f"**{savings:,.1f} 백만원 ({savings/worst_total*100:.1f}%) 절감**합니다."
        )

# -----------------------------------------------------------------------------
with tab_strategy:
    chosen = st.selectbox("상세히 살펴볼 전략을 선택하세요",
                          list(all_plans.keys()), index=0)
    df = all_plans[chosen]
    st.markdown(f"#### {chosen} 상세")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총비용",      f"{df['월총비용'].sum()/1_000:,.1f} 백만원")
    c2.metric("총 정규생산", f"{df['정규생산'].sum():,.0f} 대")
    c3.metric("총 잔업생산", f"{df['잔업생산'].sum():,.0f} 대")
    c4.metric("총 외주",     f"{df['외주'].sum():,.0f} 대")

    st.plotly_chart(fig_demand_supply(df),
                    use_container_width=True, key=f"strat_ds_{chosen}")
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(fig_workforce(df),
                        use_container_width=True, key=f"strat_wf_{chosen}")
    with cB:
        st.plotly_chart(fig_capacity_utilization(df, params),
                        use_container_width=True, key=f"strat_cap_{chosen}")
    st.plotly_chart(fig_cost_breakdown(df),
                    use_container_width=True, key=f"strat_cost_{chosen}")

    st.markdown("##### 월별 계획표")
    show_cols = ["월","수요","인력","고용","해고","정규생산","잔업시간",
                 "잔업생산","외주","총공급","기말재고","결품","월총비용"]
    fmt_df = df[show_cols].copy()
    fmt_df["월총비용"] = (fmt_df["월총비용"] / 1_000).round(2)
    fmt_df = fmt_df.rename(columns={"월총비용": "월총비용(백만원)"})
    st.dataframe(fmt_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
with tab_custom:
    st.markdown(
        "각 월의 **인력 / 잔업생산(단위) / 외주**를 직접 조정해 자기만의 시나리오를 만들고, "
        "표준 전략·최적해와 비용을 비교해 보세요."
    )

    seed_name = st.selectbox("시드(초기값) 전략 선택",
                             list(all_plans.keys()), index=0)
    seed = all_plans[seed_name]

    seed_key = f"{seed_name}_{len(demand)}"
    if "custom_seed_key" not in st.session_state or \
       st.session_state.custom_seed_key != seed_key:
        st.session_state.custom_df = pd.DataFrame({
            "월": get_months(len(demand)),
            "인력": [int(round(x)) for x in seed["인력"].tolist()],
            "잔업생산": [int(round(x)) for x in seed["잔업생산"].tolist()],
            "외주": [int(round(x)) for x in seed["외주"].tolist()],
        })
        st.session_state.custom_seed_key = seed_key

    edited = st.data_editor(
        st.session_state.custom_df,
        use_container_width=True, hide_index=True, num_rows="fixed",
        column_config={
            "월":       st.column_config.TextColumn(disabled=True),
            "인력":     st.column_config.NumberColumn(min_value=0, step=1),
            "잔업생산": st.column_config.NumberColumn(min_value=0, step=10),
            "외주":     st.column_config.NumberColumn(min_value=0, step=10),
        },
        key=f"custom_editor_{seed_key}",
    )
    st.session_state.custom_df = edited

    custom_df = plan_custom(
        demand, params,
        edited["인력"].tolist(),
        edited["잔업생산"].tolist(),
        edited["외주"].tolist(),
    )

    cmp_cols = st.columns(len(all_plans) + 1)
    cmp_cols[0].metric("Custom 총비용",
                       f"{custom_df['월총비용'].sum()/1_000:,.1f} 백만원")
    for i, (n, df) in enumerate(all_plans.items()):
        diff = (custom_df['월총비용'].sum() - df['월총비용'].sum()) / 1_000
        cmp_cols[i+1].metric(f"vs {n}", f"{diff:+,.1f} 백만원",
                             delta_color="inverse")

    st.plotly_chart(fig_demand_supply(custom_df),
                    use_container_width=True, key="custom_ds")
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(fig_workforce(custom_df),
                        use_container_width=True, key="custom_wf")
    with cB:
        st.plotly_chart(fig_cost_breakdown(custom_df),
                        use_container_width=True, key="custom_cost")

    plans_with_custom = {**all_plans, "Custom": custom_df}
    st.plotly_chart(fig_strategy_compare_cost(plans_with_custom),
                    use_container_width=True, key="custom_compare")

# -----------------------------------------------------------------------------
with tab_data:
    st.markdown("### 결과 다운로드")
    options = list(all_plans.keys()) + ["Custom (현재 시나리오)"]
    chosen = st.selectbox("내보낼 전략", options, index=0)

    if chosen.startswith("Custom"):
        cdf = st.session_state.get("custom_df")
        if cdf is None:
            st.warning("먼저 'Custom 시나리오' 탭에서 시나리오를 만들어 주세요.")
            export_df = None
        else:
            export_df = plan_custom(
                demand, params,
                cdf["인력"].tolist(),
                cdf["잔업생산"].tolist(),
                cdf["외주"].tolist(),
            )
    else:
        export_df = all_plans[chosen]

    if export_df is not None:
        st.dataframe(export_df, use_container_width=True, hide_index=True)

        csv = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📄 CSV 다운로드", csv,
                           file_name=f"APP_{chosen}.csv", mime="text/csv")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            for n, d in all_plans.items():
                d.to_excel(xw, sheet_name=n, index=False)
            if chosen.startswith("Custom") and export_df is not None:
                export_df.to_excel(xw, sheet_name="Custom", index=False)
        st.download_button("📊 전체 전략 Excel 다운로드", buf.getvalue(),
                           file_name="APP_all_strategies.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -----------------------------------------------------------------------------
with tab_help:
    st.markdown(
        "### 강의록 모델 (Pyomo LP/IP)\n\n"
        "**결정변수**: W_t (인력), H_t (고용), L_t (해고), P_t (생산), "
        "I_t (재고), S_t (부재고), C_t (외주), O_t (잔업시간)\n\n"
        "**목적함수**: Z = Σ ( 640·W_t + 6·O_t + 300·H_t + 500·L_t + "
        "2·I_t + 5·S_t + 10·P_t + 30·C_t )  [천원]\n\n"
        "**제약조건**\n"
        "- 노동력: W_t = W_{t-1} + H_t - L_t\n"
        "- 생산능력: P_t ≤ 40·W_t + 0.25·O_t\n"
        "- 재고균형: I_t = I_{t-1} + P_t + C_t - D_t - S_{t-1} + S_t\n"
        "- 잔업한도: O_t ≤ 10·W_t\n"
        "- 초기/최종: W_0 = 80, I_0 = 1000, I_N ≥ 500, S_N = 0\n\n"
        "> 본 앱은 PuLP(CBC solver)로 동일한 LP/MIP 모델을 풀이합니다. "
        "Pyomo와 수학적으로 동등하며, 결과 비용이 강의록 "
        "(LP=422,275천원, IP=422,660천원)과 일치합니다.\n\n"
        "---\n\n"
        "### 휴리스틱 전략\n"
        "- **Level (평준화)**: 평균 수요 충족 인력으로 일정 유지, 재고로 변동 흡수\n"
        "- **Chase (추적)**: 매월 수요에 맞춰 인력 조정, 재고 최소\n"
        "- **Mixed (혼합)**: 기준 인력 + 잔업/외주로 변동 흡수\n"
        "- **Custom**: 사용자가 직접 인력/잔업/외주 입력\n"
    )


st.caption("© 2026 산업공학 프로젝트 — 강의록 「스마트제조 06」 기반 / "
           "Pyomo 모델은 PuLP-CBC로 동등 구현")
