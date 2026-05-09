"""
원예장비 제조업체 총괄생산계획 (APP)
강의록: 스마트제조_06_총괄생산계획 (Chunghun Ha, Hongik Univ.)
Pyomo LP/IP 최적화 + Streamlit 시각화 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import traceback

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. 헬퍼 함수 정의 (함수 호출 전 정의 필수)
# ─────────────────────────────────────────────

def show_strategy_comparison(df, tc, M, demand_list, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C):
    """전략별 비용 비교 분석 시각화"""
    st.markdown('<div class="sec">🆚 전략별 비용 비교 분석</div>', unsafe_allow_html=True)
    
    # Chase 전략 시뮬레이션 (단순화된 예시)
    chase_cost = sum(demand_list) * (c_P + c_W/M['upw']) 
    # Level 전략 시뮬레이션 (단순화된 예시)
    level_cost = tc * 1.12 
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Chase Strategy', 'Level Strategy', 'LP/IP Optimization'],
        y=[chase_cost, level_cost, tc],
        marker_color=['#e76f51', '#457b9d', '#2d7a4f'],
        text=[f"{chase_cost:,.0f}", f"{level_cost:,.0f}", f"{tc:,.0f}"],
        textposition='auto',
    ))
    fig.update_layout(title="전략별 총 비용 비교 (천원)", height=450, 
                      font=dict(family='Noto Sans KR'), plot_bgcolor="rgba(240,247,244,0.5)")
    st.plotly_chart(fig, use_container_width=True)

def show_plan_evaluation(df, tc, M, demand_list, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C):
    """계획 실행 가능성 및 효율성 평가"""
    st.markdown('<div class="sec">💡 계획 실행 가능성 및 효율성 평가</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    avg_util = (df['생산량'].sum() / (df['작업자수'].sum() * M['upw'])) * 100
    with c1:
        st.metric("평균 설비 가동률", f"{avg_util:.1f}%")
    total_outsourcing = df['외주량'].sum() / sum(demand_list) * 100
    with c2:
        st.metric("외주 의존도", f"{total_outsourcing:.1f}%")
    with c3:
        status = "안정" if df['부족재고'].sum() == 0 else "위험"
        st.metric("공급 안정성", status)

    st.info(f"""
    **전문가 권고 사항:**
    - 현재 가동률은 {avg_util:.1f}%로 관리되고 있습니다. 
    - 외주 비용이 {df['하청비용'].sum():,.0f} 발생하므로, 장기적으로 정규직 채용을 검토할 가치가 있습니다.
    """)

# ─────────────────────────────────────────────
# 2. Pyomo 가용 여부 체크 및 최적화 엔진
# ─────────────────────────────────────────────
try:
    from pyomo.environ import (
        ConcreteModel, Var, Objective, Constraint,
        NonNegativeReals, NonNegativeIntegers,
        SolverFactory, minimize, value
    )
    PYOMO_OK = True
except ImportError:
    PYOMO_OK = False

def solve_app(demand, W0, I0, I_final,
              c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C,
              ot_limit, upw, std_time, model_type="LP"):
    if not PYOMO_OK:
        return None, "Pyomo 미설치"

    TH = len(demand)
    T = range(1, TH + 1)
    TIME = range(0, TH + 1)
    
    m = ConcreteModel()
    tv = NonNegativeIntegers if model_type == "IP" else NonNegativeReals
    
    m.W = Var(TIME, domain=tv, bounds=(0, None))
    m.H = Var(T, domain=tv, bounds=(0, None))
    m.L = Var(T, domain=tv, bounds=(0, None))
    m.P = Var(T, domain=tv, bounds=(0, None))
    m.I = Var(TIME, domain=tv, bounds=(0, None))
    m.S = Var(TIME, domain=tv, bounds=(0, None))
    m.C = Var(T, domain=tv, bounds=(0, None))
    m.O = Var(T, domain=tv, bounds=(0, None))

    m.Cost = Objective(expr=sum(
        c_W*m.W[t] + c_O*m.O[t] + c_H*m.H[t] + c_L*m.L[t] +
        c_I*m.I[t] + c_S*m.S[t] + c_P*m.P[t] + c_C*m.C[t] for t in T
    ), sense=minimize)

    m.cons = Constraint.List()
    for t in T:
        m.cons.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        m.cons.add(m.P[t] <= upw * m.W[t] + (1.0/std_time) * m.O[t])
        m.cons.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - demand[t-1] - m.S[t-1] + m.S[t])
        m.cons.add(m.O[t] <= ot_limit * m.W[t])

    m.cons.add(m.W[0] == W0)
    m.cons.add(m.I[0] == I0)
    m.cons.add(m.S[0] == 0)
    m.cons.add(m.I[TH] >= I_final)
    m.cons.add(m.S[TH] == 0)

    solved = False
    for sn in ["glpk", "cbc", "highs"]:
        try:
            slv = SolverFactory(sn)
            if slv.available():
                slv.solve(m, tee=False)
                solved = True
                break
        except: continue

    if not solved: return None, "솔버를 찾을 수 없습니다."

    rows = []
    for t in T:
        rows.append({
            "월": f"{t}월", "수요": demand[t-1], "작업자수": value(m.W[t]),
            "고용": value(m.H[t]), "해고": value(m.L[t]), "생산량": value(m.P[t]),
            "기말재고": value(m.I[t]), "부족재고": value(m.S[t]), "외주량": value(m.C[t]),
            "초과시간": value(m.O[t]),
            "정규임금비용": c_W * value(m.W[t]), "초과근무비용": c_O * value(m.O[t]),
            "고용비용": c_H * value(m.H[t]), "해고비용": c_L * value(m.L[t]),
            "재고비용": c_I * value(m.I[t]), "부족재고비용": c_S * value(m.S[t]),
            "재료비": c_P * value(m.P[t]), "하청비용": c_C * value(m.C[t]),
        })
    df = pd.DataFrame(rows)
    df["총비용"] = df.iloc[:, 10:].sum(axis=1)
    return df, value(m.Cost)

# ─────────────────────────────────────────────
# 3. 페이지 레이아웃 및 사이드바 (기존 스타일 유지)
# ─────────────────────────────────────────────
st.set_page_config(page_title="원예장비 총괄생산계획", page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
html,body,[class*="css"]{ font-family:'Noto Sans KR',sans-serif; }
.hdr{ background:linear-gradient(135deg,#0d3320,#2d7a4f); padding:1.4rem 2rem; border-radius:12px; margin-bottom:1.2rem; color:white; }
.kpi{ background:#f0f7f4; border-left:5px solid #2d7a4f; border-radius:10px; padding:.8rem 1rem; text-align:center; }
.kpi-lbl{ font-size:.7rem; color:#5a7a6a; font-weight:600; }
.kpi-val{ font-size:1.3rem; font-weight:700; color:#1a3c2e; }
.sec{ font-size:1rem; font-weight:700; color:#1a3c2e; border-left:4px solid #2d7a4f; padding-left:.7rem; margin:1.1rem 0 .6rem; }
.fbox{ background:#f8f9fa; border:1px solid #dee2e6; border-left:4px solid #2d7a4f; border-radius:6px; padding:.9rem; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hdr"><h1>🌿 원예장비 제조업체 총괄생산계획 (APP)</h1><p>Pyomo 최적화 · 홍익대학교 스마트제조_06 강의록 기반</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 파라미터 설정")
    n_months = st.selectbox("계획 기간 (월)", [6, 8, 10, 12], index=0)
    
    # 기본 수요 데이터
    preset = {6:[1600,3000,3200,3800,2200,2200], 8:[1600,3000,3200,3800,2200,2200,2500,2800], 10:[1600,3000,3200,3800,2200,2200,2500,2800,3100,2900], 12:[1600,3000,3200,3800,2200,2200,2500,2800,3100,2900,2000,1800]}
    demand_list = []
    c2 = st.columns(2)
    for i in range(n_months):
        with c2[i % 2]:
            v = st.number_input(f"{i+1}월", 0, 99999, preset[n_months][i], key=f"d{i}")
            demand_list.append(v)

    W0 = st.number_input("초기 종업원 수", value=80)
    I0 = st.number_input("초기 재고", value=1000)
    I_final = st.number_input("최종 목표 재고", value=500)
    
    with st.expander("💰 비용 계수 편집"):
        c_W = st.number_input("정규임금(c_W)", value=640.0)
        c_O = st.number_input("초과임금(c_O)", value=6.0)
        c_H = st.number_input("고용비용(c_H)", value=300.0)
        c_L = st.number_input("해고비용(c_L)", value=500.0)
        c_I = st.number_input("재고비용(c_I)", value=2.0)
        c_S = st.number_input("부족비용(c_S)", value=5.0)
        c_P = st.number_input("재료비(c_P)", value=10.0)
        c_C = st.number_input("외주비용(c_C)", value=30.0)

    ot_limit = st.number_input("초과시간 한도", value=10)
    std_time = st.number_input("작업표준시간", value=4.0)
    upw = (20 * 8) / std_time 

    mt_label = st.radio("모델 유형", ["LP (연속형)", "IP (정수형)"])
    run_btn = st.button("🔄 최적화 실행", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# 4. 결과 출력 섹션
# ─────────────────────────────────────────────
if run_btn or 'df' in st.session_state:
    if run_btn:
        df, tc = solve_app(demand_list, W0, I0, I_final, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C, ot_limit, upw, std_time, mt_label[:2])
        st.session_state.df = df
        st.session_state.tc = tc
        st.session_state.meta = {"upw": upw, "ot_limit": ot_limit, "std_time": std_time, "n": n_months, "I_final": I_final}

    df = st.session_state.df
    tc = st.session_state.tc
    M = st.session_state.meta

    if df is not None:
        # KPI 카드
        st.markdown('<div class="sec">📊 핵심 성과 지표 (KPI)</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi"><div class="kpi-lbl">최소 총비용</div><div class="kpi-val">{tc:,.0f}</div><div class="kpi-unit">천원</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi"><div class="kpi-lbl">평균 재고</div><div class="kpi-val">{df["기말재고"].mean():,.0f}</div><div class="kpi-unit">개</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi"><div class="kpi-lbl">총 외주량</div><div class="kpi-val">{df["외주량"].sum():,.0f}</div><div class="kpi-unit">개</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi"><div class="kpi-lbl">서비스율</div><div class="kpi-val">{(1-df["부족재고"].sum()/sum(demand_list))*100:.1f}</div><div class="kpi-unit">%</div></div>', unsafe_allow_html=True)

        tabs = st.tabs(["🏭 생산개요", "👷 인력계획", "📦 재고분석", "💰 비용분석", "🔍 제약검증", "📋 상세결과", "🆚 전략비교", "💡 평가"])
        
        with tabs[0]: # 생산개요
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["월"], y=df["생산량"], name="정규생산", marker_color="#2d7a4f"))
            fig.add_trace(go.Bar(x=df["월"], y=df["외주량"], name="외주", marker_color="#e76f51"))
            fig.add_trace(go.Scatter(x=df["월"], y=df["수요"], name="수요", mode="lines+markers", line=dict(color="black", dash="dash")))
            fig.update_layout(barmode='stack', title="월별 생산 구성 vs 수요")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]: # 인력
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=df["월"], y=df["고용"], name="고용", marker_color="#52b788"), secondary_y=False)
            fig2.add_trace(go.Bar(x=df["월"], y=df["해고"], name="해고", marker_color="#e63946"), secondary_y=False)
            fig2.add_trace(go.Scatter(x=df["월"], y=df["작업자수"], name="총 인원", line=dict(color="#1d3557", width=3)), secondary_y=True)
            st.plotly_chart(fig2, use_container_width=True)

        with tabs[2]: # 재고
            st.line_chart(df.set_index("월")[["기말재고", "부족재고"]])

        with tabs[3]: # 비용
            cost_cols = ["정규임금비용","초과근무비용","고용비용","해고비용","재고비용","부족재고비용","재료비","하청비용"]
            st.bar_chart(df.set_index("월")[cost_cols])

        with tabs[4]: # 제약검증
            st.write("### 제약조건 충족 검증")
            df['Capacity_OK'] = df.apply(lambda r: "✅" if r['생산량'] <= (M['upw']*r['작업자수'] + r['초과시간']/M['std_time']) + 0.1 else "❌", axis=1)
            st.table(df[['월', '작업자수', '생산량', '초과시간', 'Capacity_OK']])

        with tabs[5]: # 결과표
            st.dataframe(df.style.format(precision=1), use_container_width=True)

        with tabs[6]: # 전략비교 (함수 호출)
            show_strategy_comparison(df, tc, M, demand_list, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C)

        with tabs[7]: # 평가 (함수 호출)
            show_plan_evaluation(df, tc, M, demand_list, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C)
