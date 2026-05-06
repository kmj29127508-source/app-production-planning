"""
원예장비 제조업체 총괄생산계획(APP) 시스템
스마트제조 중간과제 - Streamlit 웹앱
강의록: 총괄생산계획 (Chunghun Ha, Hongik University)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint,
    NonNegativeReals, NonNegativeIntegers,
    SolverFactory, value, minimize, Suffix
)
import warnings
warnings.filterwarnings("ignore")

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="원예장비 총괄생산계획 시스템",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

/* 헤더 */
.hero {
    background: linear-gradient(135deg,#0a2e1a 0%,#0f3d28 60%,#0a2e1a 100%);
    border: 1px solid #1a6b3a;
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.4rem;
    position: relative; overflow: hidden;
}
.hero::after {
    content:'';position:absolute;top:-60px;right:-60px;
    width:220px;height:220px;
    background:radial-gradient(circle,rgba(34,197,94,.10) 0%,transparent 70%);
    pointer-events:none;
}
.hero-title { font-size:1.75rem;font-weight:700;color:#f0fdf4;margin:0 0 .25rem; letter-spacing:-.5px; }
.hero-sub   { color:#86efac;font-size:.82rem;letter-spacing:2px;text-transform:uppercase;font-weight:300; }

/* KPI */
.kpi { background:#161b22;border:1px solid #21262d;border-radius:10px;
       padding:1rem 1.3rem;margin-bottom:.7rem;transition:border-color .2s; }
.kpi:hover { border-color:#22c55e; }
.kpi-label { font-size:.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.3rem; }
.kpi-value { font-family:'IBM Plex Mono',monospace;font-size:1.55rem;font-weight:600;color:#f0fdf4; }
.kpi-sub   { font-size:.75rem;margin-top:.15rem; }
.green { color:#22c55e; } .red { color:#f87171; } .yellow { color:#fbbf24; } .blue { color:#60a5fa; }

/* 섹션 */
.sec-title {
    font-size:1rem;font-weight:600;color:#e2e8f0;
    border-bottom:2px solid #1a6b3a;padding-bottom:.4rem;margin-bottom:1rem;
}

/* 알림박스 */
.box-green { background:#0d2b1e;border-left:3px solid #22c55e;border-radius:0 8px 8px 0;
             padding:.75rem 1rem;margin:.4rem 0;font-size:.83rem;color:#bbf7d0; }
.box-yellow{ background:#2d1f0a;border-left:3px solid #f59e0b;border-radius:0 8px 8px 0;
             padding:.75rem 1rem;margin:.4rem 0;font-size:.83rem;color:#fde68a; }
.box-red   { background:#2d0f0f;border-left:3px solid #f87171;border-radius:0 8px 8px 0;
             padding:.75rem 1rem;margin:.4rem 0;font-size:.83rem;color:#fca5a5; }

/* 탭 */
.stTabs [data-baseweb="tab"] {
    background:#161b22;border:1px solid #21262d;border-radius:8px;
    color:#8b949e;padding:.35rem 1.1rem;font-size:.88rem;
}
.stTabs [aria-selected="true"] {
    background:#0f3d28 !important;color:#4ade80 !important;border-color:#1a6b3a !important;
}

/* 테이블 헤더색 */
thead tr th { background:#1a4d30 !important; color:#f0fdf4 !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Pyomo 최적화 모델 (강의록 그대로)
# ════════════════════════════════════════════════════════════════════════════
def solve_app(demand, params, model_type="LP"):
    """
    강의록 수식을 그대로 구현한 총괄생산계획 Pyomo 모델
    결정변수: W, H, L, P, I, S, C, O
    목적함수: Z = Σ(640*Wt + 6*Ot + 300*Ht + 500*Lt + 2*It + 5*St + 10*Pt + 30*Ct)
    """
    TH = len(demand)
    TIME = range(0, TH + 1)   # 0..TH  (0은 초기값용)
    T    = range(1, TH + 1)   # 1..TH  (실제 기간)

    p = params  # 파라미터 딕셔너리

    # 정규시간 노동비 = 임금단가 × 작업시간/일 × 작업일수/월
    reg_labor_cost = p["wage_reg"] * p["hours_per_day"] * p["days_per_month"]  # 4×8×20=640

    # 변수 타입
    domain = NonNegativeIntegers if model_type == "IP" else NonNegativeReals

    m = ConcreteModel()

    # ── 결정변수 ──
    m.W = Var(TIME, domain=domain)   # 종업원 수
    m.H = Var(TIME, domain=domain)   # 고용 인원
    m.L = Var(TIME, domain=domain)   # 해고 인원
    m.P = Var(TIME, domain=domain)   # 생산량
    m.I = Var(TIME, domain=domain)   # 재고
    m.S = Var(TIME, domain=domain)   # 부족재고(backlog)
    m.C = Var(TIME, domain=domain)   # 외주(하청)
    m.O = Var(TIME, domain=domain)   # 초과근무 총시간

    # ── 목적함수 ──
    # Z = Σ(640Wt + 6Ot + 300Ht + 500Lt + 2It + 5St + 10Pt + 30Ct)
    m.Cost = Objective(
        expr=sum(
            reg_labor_cost * m.W[t]
            + p["wage_ot"]      * m.O[t]
            + p["hire_cost"]    * m.H[t]
            + p["fire_cost"]    * m.L[t]
            + p["inv_cost"]     * m.I[t]
            + p["back_cost"]    * m.S[t]
            + p["material"]     * m.P[t]
            + p["sub_cost"]     * m.C[t]
            for t in T
        ),
        sense=minimize
    )

    # ── 제약조건 ──
    # 노동력: Wt = W(t-1) + Ht - Lt
    m.labor = Constraint(T, rule=lambda m, t:
        m.W[t] == m.W[t-1] + m.H[t] - m.L[t])

    # 생산능력: Pt ≤ 40Wt + Ot/4
    #   (규정시간 최대생산 = 1/4 ea/hr × 8 hr/day × 20 day/mon × Wt = 40Wt)
    #   (초과시간 최대생산 = Ot × 1/4)
    cap_per_worker = (1 / p["std_time"]) * p["hours_per_day"] * p["days_per_month"]
    m.capacity = Constraint(T, rule=lambda m, t:
        m.P[t] <= cap_per_worker * m.W[t] + (1/p["std_time"]) * m.O[t])

    # 재고균형: It = I(t-1) + Pt + Ct - Dt - S(t-1) + St
    D = [0] + list(demand)   # D[0]=0, D[1..TH]
    m.inventory = Constraint(T, rule=lambda m, t:
        m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t] - m.S[t-1] + m.S[t])

    # 초과근무 제한: Ot ≤ 10*Wt
    m.overtime = Constraint(T, rule=lambda m, t:
        m.O[t] <= p["max_ot_per_person"] * m.W[t])

    # ── 초기/최종 조건 ──
    m.W0 = Constraint(rule=m.W[0] == p["init_workers"])
    m.I0 = Constraint(rule=m.I[0] == p["init_inv"])
    m.S0 = Constraint(rule=m.S[0] == 0)
    m.last_inventory = Constraint(rule=m.I[TH] >= p["final_inv"])
    m.last_shortage   = Constraint(rule=m.S[TH] == 0)

    # ── 풀기 ──
    solver = SolverFactory("glpk")
    result = solver.solve(m, tee=False)

    status = str(result.solver.termination_condition)
    if status != "optimal":
        return None, status

    # ── 결과 추출 ──
    rows = []
    for t in T:
        rows.append({
            "기간": f"{t}월",
            "수요(D)":    D[t],
            "작업자(W)":  round(value(m.W[t]), 2),
            "고용(H)":    round(value(m.H[t]), 2),
            "해고(L)":    round(value(m.L[t]), 2),
            "생산량(P)":  round(value(m.P[t]), 2),
            "재고(I)":    round(value(m.I[t]), 2),
            "부족재고(S)":round(value(m.S[t]), 2),
            "외주(C)":    round(value(m.C[t]), 2),
            "초과근무(O)":round(value(m.O[t]), 2),
        })
    df = pd.DataFrame(rows)

    # 비용 상세 분해
    cost_detail = {}
    for t in T:
        cost_detail[f"{t}월"] = {
            "정규노동비": round(reg_labor_cost * value(m.W[t])),
            "초과근무비": round(p["wage_ot"] * value(m.O[t])),
            "고용비":     round(p["hire_cost"] * value(m.H[t])),
            "해고비":     round(p["fire_cost"] * value(m.L[t])),
            "재고유지비": round(p["inv_cost"]  * value(m.I[t])),
            "부족재고비": round(p["back_cost"] * value(m.S[t])),
            "재료비":     round(p["material"]  * value(m.P[t])),
            "하청비":     round(p["sub_cost"]  * value(m.C[t])),
        }

    total_cost = round(value(m.Cost))
    return {
        "df": df,
        "total_cost": total_cost,
        "cost_detail": cost_detail,
        "status": status,
        "model": m,
        "TH": TH,
        "D": D,
    }, "optimal"


# ════════════════════════════════════════════════════════════════════════════
#  사이드바 – 파라미터 입력
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 총괄생산계획")
    st.markdown("**원예장비 제조업체**")
    st.divider()

    st.markdown("### 📅 수요 예측 (개/월)")
    n_periods = st.selectbox("계획 기간(월)", [6, 7, 8, 9, 10, 12], index=0)

    default_demand = [1600, 3000, 3200, 3800, 2200, 2200]
    demand_inputs = []
    cols_d = st.columns(2)
    for i in range(n_periods):
        default = default_demand[i] if i < len(default_demand) else 2000
        with cols_d[i % 2]:
            v = st.number_input(f"{i+1}월", min_value=0, max_value=20000,
                                value=default, step=100, key=f"d{i}")
        demand_inputs.append(v)

    st.divider()
    st.markdown("### ⚙️ 파라미터 설정")

    with st.expander("💰 비용 파라미터 (천원)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            sale_price  = st.number_input("판매단가",    value=40,  step=1)
            material    = st.number_input("재료비/개",   value=10,  step=1)
            inv_cost    = st.number_input("재고유지비/개/월", value=2, step=1)
            back_cost   = st.number_input("부재고비/개/월",  value=5, step=1)
        with c2:
            hire_cost   = st.number_input("고용비/인",   value=300, step=10)
            fire_cost   = st.number_input("해고비/인",   value=500, step=10)
            sub_cost    = st.number_input("하청비/개",   value=30,  step=1)
            wage_reg    = st.number_input("정규임금(천원/시간)", value=4, step=1)
        wage_ot = st.number_input("초과근무임금(천원/시간)", value=6, step=1)

    with st.expander("👷 인력 & 생산 파라미터"):
        c3, c4 = st.columns(2)
        with c3:
            init_workers = st.number_input("초기 작업자(명)", value=80, step=1)
            init_inv     = st.number_input("초기 재고(개)",   value=1000, step=100)
            final_inv    = st.number_input("최종 재고(개)",   value=500, step=100)
        with c4:
            days_per_month    = st.number_input("작업일수/월", value=20, step=1)
            hours_per_day     = st.number_input("작업시간/일", value=8,  step=1)
            max_ot_per_person = st.number_input("최대초과시간/인/월", value=10, step=1)
        std_time = st.number_input("작업표준시간(시간/개)", value=4.0, step=0.5)

    st.divider()
    model_type = st.radio("모델 타입", ["LP (연속)", "IP (정수)"],
                          captions=["빠른 계산", "현실적 정수해"])
    mtype = "LP" if model_type.startswith("LP") else "IP"

    run_btn = st.button("🚀 최적화 실행", use_container_width=True, type="primary")

# 파라미터 딕셔너리
params = dict(
    wage_reg=wage_reg, wage_ot=wage_ot,
    hire_cost=hire_cost, fire_cost=fire_cost,
    inv_cost=inv_cost, back_cost=back_cost,
    material=material, sub_cost=sub_cost,
    sale_price=sale_price,
    init_workers=init_workers, init_inv=init_inv, final_inv=final_inv,
    days_per_month=days_per_month, hours_per_day=hours_per_day,
    max_ot_per_person=max_ot_per_person, std_time=std_time,
)


# ════════════════════════════════════════════════════════════════════════════
#  메인 화면
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">🌿 원예장비 제조업체 총괄생산계획 시스템</div>
  <div class="hero-sub">Smart Manufacturing · Aggregate Production Planning · Pyomo LP/IP Optimization</div>
</div>
""", unsafe_allow_html=True)

# 첫 실행 안내
if "result" not in st.session_state:
    st.info("👈 왼쪽 사이드바에서 수요와 파라미터를 설정한 후 **'🚀 최적화 실행'** 버튼을 눌러주세요.")

    # 모델 개요 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 📌 총괄생산계획이란?
        - 제품군 단위의 **중장기 생산계획** 수립
        - 생산용량(설비, 인력) 조정 및 예산 확보
        - **이익 최대화 = 비용 최소화** 목적
        """)
    with col2:
        st.markdown("""
        #### 🔧 결정변수 (8개)
        | 변수 | 의미 |
        |------|------|
        | Wₜ | t월 종업원 수 |
        | Hₜ | t월 고용 인원 |
        | Lₜ | t월 해고 인원 |
        | Pₜ | t월 생산량 |
        | Iₜ | t월 말 재고 |
        | Sₜ | t월 말 부족재고 |
        | Cₜ | t월 하청 수량 |
        | Oₜ | t월 초과근무 시간 |
        """)
    with col3:
        st.markdown("""
        #### 📐 목적함수
        ```
        Z = Σ(640Wt + 6Ot
             + 300Ht + 500Lt
             + 2It  + 5St
             + 10Pt + 30Ct)
        ```
        #### 📋 제약조건
        - 노동력 연속성: Wt = W(t-1)+Ht-Lt
        - 생산능력: Pt ≤ 40Wt + Ot/4
        - 재고균형: It = I(t-1)+Pt+Ct-Dt...
        - 초과근무: Ot ≤ 10×Wt
        """)
    st.stop()

# ── 최적화 실행 ──────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("🔄 Pyomo GLPK 최적화 실행 중..."):
        res, status = solve_app(demand_inputs, params, mtype)
    if status != "optimal":
        st.error(f"❌ 최적해를 찾을 수 없습니다. 상태: {status}")
        st.stop()
    st.session_state["result"] = res
    st.session_state["demand"] = demand_inputs
    st.session_state["params"] = params
    st.session_state["mtype"]  = mtype
    st.success("✅ 최적화 완료!")

res    = st.session_state.get("result")
demand = st.session_state.get("demand", demand_inputs)
mtype_used = st.session_state.get("mtype", mtype)

if res is None:
    st.stop()

df    = res["df"]
TH    = res["TH"]
D     = res["D"]
total = res["total_cost"]
cd    = res["cost_detail"]   # 비용 상세

months = [f"{t}월" for t in range(1, TH+1)]

# ════════════════════════════════════════════════════════════════════════════
#  KPI 요약
# ════════════════════════════════════════════════════════════════════════════
total_demand   = sum(demand)
total_prod     = df["생산량(P)"].sum()
total_sub      = df["외주(C)"].sum()
total_shortage = df["부족재고(S)"].sum()
total_revenue  = total_demand * params["sale_price"]
profit         = total_revenue - total

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "최소 총비용",   f"{total:,}", "천원", "green"),
    (k2, "총 이익(추정)", f"{profit:,}", "천원", "green" if profit > 0 else "red"),
    (k3, "총 생산량",     f"{int(total_prod):,}", "개", "blue"),
    (k4, "총 외주량",     f"{int(total_sub):,}", "개", "yellow" if total_sub > 0 else "green"),
    (k5, "누적 부족재고", f"{total_shortage:.1f}", "개", "red" if total_shortage > 0 else "green"),
]
for col, label, val, unit, color in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {color}">{val}</div>
          <div class="kpi-sub {color}">{unit} · {mtype_used}</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  탭 구성
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 생산계획 결과표",
    "📈 수요·생산·재고",
    "👷 인력 계획",
    "💰 비용 분석",
    "🔍 적절성 평가",
])

COLORS = {
    "demand":   "#f87171",
    "prod":     "#4ade80",
    "inv":      "#60a5fa",
    "shortage": "#fb923c",
    "worker":   "#c084fc",
    "hire":     "#34d399",
    "fire":     "#f87171",
    "ot":       "#fbbf24",
    "sub":      "#e879f9",
}
CHART_BG   = "rgba(0,0,0,0)"
PAPER_BG   = "rgba(15,17,23,0)"
FONT_COLOR = "#e2e8f0"
GRID_COLOR = "#21262d"


def base_layout(title="", h=420):
    return dict(
        title=dict(text=title, font=dict(color=FONT_COLOR, size=14)),
        plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, family="Noto Sans KR"),
        height=h,
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(22,27,34,.8)", bordercolor="#21262d", borderwidth=1),
        margin=dict(l=50, r=30, t=50, b=40),
    )


# ── TAB 1: 결과표 ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-title">📋 총괄생산계획 결과표</div>', unsafe_allow_html=True)

    # 스타일된 데이터프레임
    display_df = df.copy()
    st.dataframe(
        display_df.style
            .format({col: "{:.1f}" for col in display_df.columns if col != "기간"})
            .background_gradient(subset=["생산량(P)"], cmap="Greens")
            .background_gradient(subset=["재고(I)"],   cmap="Blues")
            .background_gradient(subset=["부족재고(S)"], cmap="Reds")
            .set_properties(**{"font-size": "13px"}),
        use_container_width=True, height=300
    )

    # 비용 상세표
    st.markdown('<div class="sec-title" style="margin-top:1.2rem">💸 월별 비용 상세 (천원)</div>',
                unsafe_allow_html=True)
    cost_df = pd.DataFrame(cd).T
    cost_df["합계"] = cost_df.sum(axis=1)

    st.dataframe(
        cost_df.style
            .format("{:,.0f}")
            .background_gradient(subset=["합계"], cmap="YlOrRd")
            .set_properties(**{"font-size": "13px"}),
        use_container_width=True
    )

    # 목적함수 수식 표시
    reg_lc = params["wage_reg"] * params["hours_per_day"] * params["days_per_month"]
    st.markdown(f"""
    <div class="box-green">
    📐 <b>목적함수 (비용 최소화)</b>: 
    Z = Σ({reg_lc}·Wt + {params['wage_ot']}·Ot + {params['hire_cost']}·Ht + {params['fire_cost']}·Lt 
    + {params['inv_cost']}·It + {params['back_cost']}·St + {params['material']}·Pt + {params['sub_cost']}·Ct)
    &nbsp;→&nbsp; <b>최소 총비용 = {total:,} 천원</b>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 2: 수요·생산·재고 ────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-title">📈 수요 · 생산 · 재고 현황</div>', unsafe_allow_html=True)

    # 수요 vs 생산 vs 외주
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=months, y=df["수요(D)"], name="수요",
                          marker_color=COLORS["demand"], opacity=.8))
    fig1.add_trace(go.Bar(x=months, y=df["생산량(P)"], name="생산량",
                          marker_color=COLORS["prod"], opacity=.85))
    fig1.add_trace(go.Bar(x=months, y=df["외주(C)"], name="외주",
                          marker_color=COLORS["sub"], opacity=.8))
    fig1.add_trace(go.Scatter(x=months, y=df["수요(D)"], name="수요선",
                              mode="lines+markers",
                              line=dict(color=COLORS["demand"], width=2.5, dash="dot"),
                              marker=dict(size=7)))
    fig1.update_layout(barmode="group", **base_layout("수요 vs 생산량 vs 외주량"))
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)

    # 재고 추이
    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=months, y=df["재고(I)"],
            fill="tozeroy", fillcolor="rgba(96,165,250,.15)",
            line=dict(color=COLORS["inv"], width=2.5),
            mode="lines+markers", marker=dict(size=7), name="기말재고"
        ))
        fig2.add_trace(go.Scatter(
            x=months, y=df["부족재고(S)"],
            fill="tozeroy", fillcolor="rgba(248,113,113,.15)",
            line=dict(color=COLORS["shortage"], width=2.5),
            mode="lines+markers", marker=dict(size=7), name="부족재고"
        ))
        # 최종 재고 목표선
        fig2.add_hline(y=params["final_inv"], line_dash="dash",
                       line_color="#fbbf24", annotation_text=f"목표재고 {params['final_inv']}개")
        fig2.update_layout(**base_layout("월별 재고 & 부족재고 추이", h=360))
        st.plotly_chart(fig2, use_container_width=True)

    # 누적 공급-수요 비교
    with col_b:
        cum_demand = np.cumsum(df["수요(D)"])
        cum_supply = np.cumsum(df["생산량(P)"] + df["외주(C)"])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=months, y=cum_demand,
                                  name="누적 수요",
                                  line=dict(color=COLORS["demand"], width=2.5),
                                  mode="lines+markers"))
        fig3.add_trace(go.Scatter(x=months, y=cum_supply,
                                  name="누적 공급(생산+외주)",
                                  line=dict(color=COLORS["prod"], width=2.5),
                                  mode="lines+markers"))
        fig3.add_trace(go.Scatter(
            x=months + months[::-1],
            y=list(cum_supply) + list(cum_demand[::-1]),
            fill="toself", fillcolor="rgba(34,197,94,.07)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False
        ))
        fig3.update_layout(**base_layout("누적 수요 vs 누적 공급", h=360))
        st.plotly_chart(fig3, use_container_width=True)

    # 초과근무 현황
    st.markdown('<div class="sec-title">⏰ 초과근무 현황</div>', unsafe_allow_html=True)
    fig_ot = go.Figure()
    ot_max = [params["max_ot_per_person"] * w for w in df["작업자(W)"]]
    fig_ot.add_trace(go.Bar(x=months, y=df["초과근무(O)"], name="실제 초과근무",
                            marker_color=COLORS["ot"]))
    fig_ot.add_trace(go.Scatter(x=months, y=ot_max, name="최대 허용 초과근무",
                                line=dict(color="#f87171", dash="dash", width=2),
                                mode="lines"))
    fig_ot.update_layout(barmode="overlay", **base_layout("월별 초과근무 시간 (Hr/Month)", h=320))
    st.plotly_chart(fig_ot, use_container_width=True)


# ── TAB 3: 인력 계획 ────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-title">👷 인력 계획 Dashboard</div>', unsafe_allow_html=True)

    col_w1, col_w2 = st.columns(2)

    # 작업자 수 추이
    with col_w1:
        w_all = [params["init_workers"]] + list(df["작업자(W)"])
        months_all = ["초기"] + months
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(
            x=months_all, y=w_all,
            fill="tozeroy", fillcolor="rgba(192,132,252,.12)",
            line=dict(color=COLORS["worker"], width=2.5),
            mode="lines+markers+text",
            marker=dict(size=8),
            text=[str(int(v)) for v in w_all],
            textposition="top center",
            name="작업자 수(명)"
        ))
        fig_w.update_layout(**base_layout("월별 작업자 수 변화", h=380))
        st.plotly_chart(fig_w, use_container_width=True)

    # 고용 / 해고
    with col_w2:
        fig_hl = go.Figure()
        fig_hl.add_trace(go.Bar(x=months, y=df["고용(H)"],
                                name="고용", marker_color=COLORS["hire"]))
        fig_hl.add_trace(go.Bar(x=months, y=[-v for v in df["해고(L)"]],
                                name="해고", marker_color=COLORS["fire"]))
        net = df["고용(H)"] - df["해고(L)"]
        fig_hl.add_trace(go.Scatter(x=months, y=net, name="순 인력변동",
                                    mode="lines+markers",
                                    line=dict(color="#fbbf24", width=2.5),
                                    marker=dict(size=7)))
        fig_hl.add_hline(y=0, line_color="#8b949e", line_dash="solid", line_width=1)
        fig_hl.update_layout(**base_layout("월별 고용 / 해고 현황", h=380))
        st.plotly_chart(fig_hl, use_container_width=True)

    # 생산 능력 활용률
    st.markdown('<div class="sec-title">⚡ 생산능력 활용률</div>', unsafe_allow_html=True)
    cap_per_worker = (1/params["std_time"]) * params["hours_per_day"] * params["days_per_month"]
    max_reg = [cap_per_worker * w for w in df["작업자(W)"]]
    max_tot = [cap_per_worker * w + (1/params["std_time"]) * o
               for w, o in zip(df["작업자(W)"], df["초과근무(O)"])]
    util_reg = [min(p/m*100, 100) if m > 0 else 0 for p, m in zip(df["생산량(P)"], max_reg)]
    util_tot = [p/m*100 if m > 0 else 0 for p, m in zip(df["생산량(P)"], max_tot)]

    fig_util = go.Figure()
    fig_util.add_trace(go.Bar(x=months, y=max_reg, name="정규시간 최대생산",
                              marker_color="rgba(74,222,128,.3)"))
    fig_util.add_trace(go.Bar(x=months, y=max_tot, name="초과포함 최대생산",
                              marker_color="rgba(74,222,128,.15)"))
    fig_util.add_trace(go.Scatter(x=months, y=df["생산량(P)"], name="실제 생산량",
                                  mode="lines+markers",
                                  line=dict(color=COLORS["prod"], width=2.5),
                                  marker=dict(size=8)))
    fig_util.update_layout(barmode="overlay", **base_layout("생산능력 활용 현황", h=360))
    st.plotly_chart(fig_util, use_container_width=True)

    # 활용률 게이지
    cols_g = st.columns(min(TH, 6))
    for i, (col, m_) in enumerate(zip(cols_g, months)):
        with col:
            util = util_reg[i]
            color = "#22c55e" if util <= 90 else "#f59e0b" if util <= 100 else "#f87171"
            st.metric(label=m_, value=f"{util:.1f}%", delta=f"{util_tot[i]:.1f}% (초과포함)")


# ── TAB 4: 비용 분석 ────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-title">💰 비용 구조 분석</div>', unsafe_allow_html=True)

    cost_df2 = pd.DataFrame(cd).T
    cost_categories = list(cost_df2.columns)
    cat_colors = ["#4ade80","#fbbf24","#34d399","#f87171","#60a5fa","#fb923c","#e879f9","#94a3b8"]

    col_p1, col_p2 = st.columns(2)

    # 파이차트 – 전체 비용 구성
    with col_p1:
        total_by_cat = cost_df2.sum()
        fig_pie = go.Figure(go.Pie(
            labels=cost_categories, values=total_by_cat,
            hole=0.45,
            marker=dict(colors=cat_colors),
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            title=dict(text="전체 비용 구성 비율", font=dict(color=FONT_COLOR, size=13)),
            paper_bgcolor=PAPER_BG,
            font=dict(color=FONT_COLOR),
            height=380,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # 월별 비용 스택 바
    with col_p2:
        fig_stack = go.Figure()
        for i, cat in enumerate(cost_categories):
            fig_stack.add_trace(go.Bar(
                x=months, y=cost_df2[cat],
                name=cat, marker_color=cat_colors[i]
            ))
        fig_stack.update_layout(
            barmode="stack",
            **base_layout("월별 비용 구성 (천원)", h=380)
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    # 비용 vs 수익 워터폴
    st.markdown('<div class="sec-title">💹 수익성 분석</div>', unsafe_allow_html=True)
    revenue   = int(total_demand * params["sale_price"])
    cost_val  = int(total)
    profit_v  = revenue - cost_val

    fig_wf = go.Figure(go.Waterfall(
        name="수익 분석",
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["총 매출", "총 비용 (음수)", "순이익"],
        y=[revenue, -cost_val, 0],
        text=[f"{revenue:,}", f"-{cost_val:,}", f"{profit_v:,}"],
        textposition="outside",
        connector=dict(line=dict(color="#21262d")),
        increasing=dict(marker=dict(color="#4ade80")),
        decreasing=dict(marker=dict(color="#f87171")),
        totals=dict(marker=dict(color="#60a5fa")),
    ))
    fig_wf.update_layout(**base_layout("수익 구조 (천원)", h=360))
    st.plotly_chart(fig_wf, use_container_width=True)

    # 비용 상세 KPI
    c1c, c2c, c3c, c4c = st.columns(4)
    total_by_cat2 = cost_df2.sum()
    with c1c:
        st.metric("정규노동비", f"{int(total_by_cat2['정규노동비']):,} 천원",
                  f"{total_by_cat2['정규노동비']/total*100:.1f}%")
    with c2c:
        st.metric("재료비",    f"{int(total_by_cat2['재료비']):,} 천원",
                  f"{total_by_cat2['재료비']/total*100:.1f}%")
    with c3c:
        st.metric("재고유지비", f"{int(total_by_cat2['재고유지비']):,} 천원",
                  f"{total_by_cat2['재고유지비']/total*100:.1f}%")
    with c4c:
        v_hire_fire = total_by_cat2["고용비"] + total_by_cat2["해고비"]
        st.metric("인력조정비", f"{int(v_hire_fire):,} 천원",
                  f"{v_hire_fire/total*100:.1f}%")


# ── TAB 5: 적절성 평가 ───────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="sec-title">🔍 계획 적절성 평가 Dashboard</div>', unsafe_allow_html=True)

    issues   = []
    warnings = []
    goods    = []

    # 1. 부족재고 검사
    if df["부족재고(S)"].sum() > 0:
        months_short = [months[i] for i, v in enumerate(df["부족재고(S)"]) if v > 0]
        issues.append(f"⚠️ 부족재고 발생 ({', '.join(months_short)}) — 고객 서비스 저하 위험")
    else:
        goods.append("✅ 모든 기간 부족재고 없음 — 고객 서비스 양호")

    # 2. 최종 재고 검사
    final_actual = df["재고(I)"].iloc[-1]
    if final_actual < params["final_inv"]:
        issues.append(f"❌ 최종 재고 부족: {final_actual:.0f}개 < 목표 {params['final_inv']}개")
    else:
        goods.append(f"✅ 최종 재고 달성: {final_actual:.0f}개 ≥ 목표 {params['final_inv']}개")

    # 3. 초과근무율 검사
    for i, (o, w) in enumerate(zip(df["초과근무(O)"], df["작업자(W)"])):
        max_ot = params["max_ot_per_person"] * w
        if w > 0 and o > max_ot * 0.9:
            warnings.append(f"🟡 {months[i]} 초과근무 한계 근접 ({o:.1f}/{max_ot:.1f} hr)")

    # 4. 인력 변동 검사
    total_hire = df["고용(H)"].sum()
    total_fire = df["해고(L)"].sum()
    if total_hire > 0 or total_fire > 0:
        warnings.append(f"🟡 인력 조정 발생: 고용 {total_hire:.0f}명, 해고 {total_fire:.0f}명 — 고용비용 {int(total_hire*params['hire_cost']+total_fire*params['fire_cost']):,} 천원")

    # 5. 외주 의존도
    sub_ratio = df["외주(C)"].sum() / max(df["생산량(P)"].sum() + df["외주(C)"].sum(), 1) * 100
    if sub_ratio > 20:
        issues.append(f"⚠️ 외주 의존도 높음: {sub_ratio:.1f}% — 품질·납기 리스크")
    elif sub_ratio > 0:
        warnings.append(f"🟡 외주 사용: {sub_ratio:.1f}%")
    else:
        goods.append("✅ 외주 없음 — 자체 생산 충족")

    # 6. 평균 재고 수준
    avg_inv = df["재고(I)"].mean()
    if avg_inv > 2000:
        warnings.append(f"🟡 평균 재고 높음: {avg_inv:.0f}개 — 재고유지비 부담")
    else:
        goods.append(f"✅ 평균 재고 적정: {avg_inv:.0f}개")

    # 출력
    col_ev1, col_ev2 = st.columns([1, 2])
    with col_ev1:
        # 종합 점수
        score = max(0, 100 - len(issues)*25 - len(warnings)*10)
        color_s = "#22c55e" if score >= 80 else "#fbbf24" if score >= 60 else "#f87171"
        grade   = "우수" if score >= 80 else "보통" if score >= 60 else "개선필요"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0,1], "y": [0,1]},
            title={"text": f"계획 적절성 점수<br><span style='font-size:.85em;color:{color_s}'>{grade}</span>",
                   "font": {"color": FONT_COLOR, "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": FONT_COLOR},
                "bar":  {"color": color_s},
                "bgcolor": "#21262d",
                "steps": [
                    {"range": [0,  60], "color": "rgba(248,113,113,.15)"},
                    {"range": [60, 80], "color": "rgba(251,191,36,.15)"},
                    {"range": [80,100], "color": "rgba(34,197,94,.15)"},
                ],
                "threshold": {"line": {"color": "#f0fdf4", "width": 2},
                              "thickness": .75, "value": 80},
            },
            number={"suffix": "점", "font": {"color": FONT_COLOR}},
        ))
        fig_gauge.update_layout(paper_bgcolor=PAPER_BG, height=300,
                                margin=dict(l=30,r=30,t=60,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_ev2:
        if issues:
            for msg in issues:
                st.markdown(f'<div class="box-red">{msg}</div>', unsafe_allow_html=True)
        if warnings:
            for msg in warnings:
                st.markdown(f'<div class="box-yellow">{msg}</div>', unsafe_allow_html=True)
        if goods:
            for msg in goods:
                st.markdown(f'<div class="box-green">{msg}</div>', unsafe_allow_html=True)

    # 제약조건 충족 여부 체크테이블
    st.markdown('<div class="sec-title" style="margin-top:1.2rem">📋 제약조건 충족 현황</div>',
                unsafe_allow_html=True)
    cap_per_w = (1/params["std_time"]) * params["hours_per_day"] * params["days_per_month"]
    checks = []
    for i, row in df.iterrows():
        t = i + 1
        w = row["작업자(W)"]
        p_v = row["생산량(P)"]
        o   = row["초과근무(O)"]
        inv = row["재고(I)"]
        s_v = row["부족재고(S)"]

        cap_ok  = p_v <= cap_per_w * w + (1/params["std_time"]) * o + 0.01
        ot_ok   = o <= params["max_ot_per_person"] * w + 0.01
        inv_ok  = inv >= 0
        back_ok = s_v >= 0

        checks.append({
            "기간": f"{t}월",
            "생산능력": "✅" if cap_ok  else "❌",
            "초과근무": "✅" if ot_ok   else "❌",
            "재고≥0":   "✅" if inv_ok  else "❌",
            "부족≥0":   "✅" if back_ok else "❌",
        })

    # 최종 재고
    final_ok  = df["재고(I)"].iloc[-1] >= params["final_inv"] - 0.01
    final_s_ok= df["부족재고(S)"].iloc[-1] <= 0.01
    checks_df = pd.DataFrame(checks)
    checks_df.loc[len(checks_df)] = {
        "기간": "최종조건",
        "생산능력": "-",
        "초과근무": "-",
        "재고≥0": f"{'✅' if final_ok else '❌'} (≥{params['final_inv']})",
        "부족≥0": f"{'✅' if final_s_ok else '❌'} (=0)",
    }
    st.dataframe(checks_df, use_container_width=True, hide_index=True)

    # 감도 분석 – 수요 ±20%
    st.markdown('<div class="sec-title" style="margin-top:1.2rem">📊 감도 분석 (수요 변동 시나리오)</div>',
                unsafe_allow_html=True)
    scenarios = {"기준 (-20%)": 0.8, "기준 (0%)": 1.0, "기준 (+20%)": 1.2}
    scen_costs = {}
    for name, factor in scenarios.items():
        d_scen = [int(d * factor) for d in demand]
        res_s, st_s = solve_app(d_scen, params, mtype_used)
        scen_costs[name] = res_s["total_cost"] if res_s else None

    fig_sens = go.Figure()
    valid = {k: v for k, v in scen_costs.items() if v is not None}
    fig_sens.add_trace(go.Bar(
        x=list(valid.keys()), y=list(valid.values()),
        marker_color=["#60a5fa", "#4ade80", "#f87171"],
        text=[f"{v:,}" for v in valid.values()],
        textposition="outside",
        width=0.4,
    ))
    fig_sens.update_layout(**base_layout("수요 변동 시나리오별 최소 총비용 (천원)", h=340))
    st.plotly_chart(fig_sens, use_container_width=True)

# ── 푸터 ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#8b949e;font-size:.78rem;padding:.5rem 0;">
🌿 원예장비 제조업체 총괄생산계획 시스템 &nbsp;|&nbsp;
스마트제조 중간과제 &nbsp;|&nbsp;
Pyomo LP/IP Optimization · GLPK Solver &nbsp;|&nbsp;
Chunghun Ha, Hongik University 강의록 기반
</div>
""", unsafe_allow_html=True)
