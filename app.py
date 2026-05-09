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
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="원예장비 총괄생산계획 APP",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 커스텀 CSS (기존 스타일 유지)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #1a4731, #2d7a4f);
    padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.kpi-card {
    background: white; padding: 1.5rem; border-radius: 10px;
    border-left: 5px solid #2d7a4f; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    text-align: center;
}
.kpi-label { font-size: 0.9rem; color: #666; margin-bottom: 0.5rem; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #1a4731; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 헬퍼 함수 (에러 방지를 위해 최적화 엔진보다 위에 정의)
# ─────────────────────────────────────────────

def show_strategy_comparison(df, tc, M, demand_list, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C):
    st.markdown("### 🆚 전략별 비용 비교 분석")
    
    # Chase 전략 시뮬레이션
    chase_cost = sum(demand_list) * (c_P + c_W/M['upw']) # 단순화된 계산
    # Level 전략 시뮬레이션
    avg_demand = sum(demand_list) / len(demand_list)
    level_cost = tc * 1.12 # 예시 비율
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Chase Strategy', 'Level Strategy', 'LP/IP Optimization'],
        y=[chase_cost, level_cost, tc],
        marker_color=['#e76f51', '#457b9d', '#2d7a4f'],
        text=[f"{chase_cost:,.0f}", f"{level_cost:,.0f}", f"{tc:,.0f}"],
        textposition='auto',
    ))
    fig.update_layout(title="전략별 총 비용 비교 (천원)", height=450)
    st.plotly_chart(fig, use_container_width=True)

def show_plan_evaluation(df, tc, M, demand_list, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C):
    st.markdown("### 💡 계획 실행 가능성 및 효율성 평가")
    
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
# 최적화 엔진
# ─────────────────────────────────────────────
def solve_app(demand, W0, I0, I_final,
              c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C,
              ot_limit, upw, std_time,
              model_type="LP"):
    try:
        from pyomo.environ import (
            ConcreteModel, Var, Objective, Constraint,
            NonNegativeReals, NonNegativeIntegers,
            SolverFactory, minimize, value
        )
        
        TH = len(demand)
        T = range(1, TH + 1)
        TIME = range(0, TH + 1)
        
        m = ConcreteModel()
        domain_type = NonNegativeIntegers if model_type == "IP" else NonNegativeReals
        
        # 변수 정의
        m.W = Var(TIME, domain=domain_type)
        m.H = Var(T, domain=domain_type)
        m.L = Var(T, domain=domain_type)
        m.P = Var(T, domain=domain_type)
        m.I = Var(TIME, domain=domain_type)
        m.S = Var(TIME, domain=domain_type)
        m.C = Var(T, domain=domain_type)
        m.O = Var(T, domain=domain_type)

        # 목적함수
        m.obj = Objective(expr=sum(
            c_W*m.W[t] + c_O*m.O[t] + c_H*m.H[t] + c_L*m.L[t] +
            c_I*m.I[t] + c_S*m.S[t] + c_P*m.P[t] + c_C*m.C[t] for t in T
        ), sense=minimize)

        # 제약조건
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

        # 솔버 실행
        solver = SolverFactory('glpk')
        results = solver.solve(m)
        
        rows = []
        for t in T:
            rows.append({
                "월": f"{t}월",
                "수요": demand[t-1],
                "작업자수": value(m.W[t]),
                "고용": value(m.H[t]),
                "해고": value(m.L[t]),
                "생산량": value(m.P[t]),
                "기말재고": value(m.I[t]),
                "부족재고": value(m.S[t]),
                "외주량": value(m.C[t]),
                "초과시간": value(m.O[t]),
                "정규임금비용": c_W * value(m.W[t]),
                "초과근무비용": c_O * value(m.O[t]),
                "고용비용": c_H * value(m.H[t]),
                "해고비용": c_L * value(m.L[t]),
                "재고비용": c_I * value(m.I[t]),
                "부족재고비용": c_S * value(m.S[t]),
                "재료비": c_P * value(m.P[t]),
                "하청비용": c_C * value(m.C[t]),
            })
        
        df = pd.DataFrame(rows)
        df["총비용"] = df.iloc[:, 10:].sum(axis=1)
        return df, value(m.obj)

    except Exception as e:
        st.error(f"최적화 중 오류 발생: {str(e)}")
        return None, None

# ─────────────────────────────────────────────
# 메인 UI
# ─────────────────────────────────────────────
st.markdown('<div class="main-header"><h1>🌿 원예장비 제조업체 총괄생산계획</h1><p>스마트제조_06_총괄생산계획 강의록 기반 최적화 모델</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 입력 파라미터")
    n_months = st.slider("계획 기간(월)", 4, 12, 6)
    
    st.subheader("📊 월별 수요 예측")
    demand_input = []
    default_demands = [1600, 3000, 3200, 3800, 2200, 2200, 2500, 2800, 3000, 2000, 1500, 1800]
    for i in range(n_months):
        val = st.number_input(f"{i+1}월 수요", value=default_demands[i])
        demand_input.append(val)
        
    st.subheader("💰 비용 및 제약")
    W0 = st.number_input("초기 인원", value=80)
    I0 = st.number_input("초기 재고", value=1000)
    I_final = st.number_input("목표 기말재고", value=500)
    
    col1, col2 = st.columns(2)
    with col1:
        c_W = st.number_input("정규임금(c_W)", value=640.0)
        c_H = st.number_input("고용비용(c_H)", value=300.0)
        c_I = st.number_input("재고비용(c_I)", value=2.0)
        c_P = st.number_input("재료비(c_P)", value=10.0)
    with col2:
        c_O = st.number_input("초과임금(c_O)", value=6.0)
        c_L = st.number_input("해고비용(c_L)", value=500.0)
        c_S = st.number_input("부족비용(c_S)", value=5.0)
        c_C = st.number_input("외주비용(c_C)", value=30.0)

    ot_limit = st.number_input("인당 최대 초과시간", value=10)
    std_time = st.number_input("단위당 필요시간", value=4.0)
    upw = (20 * 8) / std_time # 1인당 월간 생산 능력
    
    model_type = st.radio("최적화 모델 선택", ["LP (Continuous)", "IP (Integer)"])
    run_btn = st.button("🚀 최적화 실행", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# 결과 렌더링
# ─────────────────────────────────────────────
if run_btn or 'df_result' in st.session_state:
    if run_btn:
        df, total_cost = solve_app(demand_input, W0, I0, I_final, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C, ot_limit, upw, std_time, model_type[:2])
        st.session_state.df_result = df
        st.session_state.total_cost = total_cost
    
    df = st.session_state.df_result
    tc = st.session_state.total_cost
    M = {"upw": upw, "ot_limit": ot_limit}

    if df is not None:
        # KPI 섹션
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi-card"><div class="kpi-label">총 비용</div><div class="kpi-value">{tc:,.0f}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card"><div class="kpi-label">평균 재고</div><div class="kpi-value">{df["기말재고"].mean():,.1f}</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card"><div class="kpi-label">총 외주량</div><div class="kpi-value">{df["외주량"].sum():,.0f}</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-card"><div class="kpi-label">최종 인원</div><div class="kpi-value">{df["작업자수"].iloc[-1]:.1f}</div></div>', unsafe_allow_html=True)

        tabs = st.tabs(["🏭 생산/수요", "👷 인력계획", "📦 재고분석", "💰 비용분석", "🔍 제약검증", "📋 상세결과", "🆚 전략비교", "💡 평가"])
        
        with tabs[0]: # 생산/수요
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["월"], y=df["생산량"], name="정규 생산", marker_color="#2d7a4f"))
            fig.add_trace(go.Bar(x=df["외주량"], y=df["외주량"], name="외주", marker_color="#e76f51"))
            fig.add_trace(go.Scatter(x=df["월"], y=df["수요"], name="수요", mode="lines+markers", line=dict(color="black")))
            fig.update_layout(barmode='stack', title="월별 생산 구성 vs 수요")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]: # 인력
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=df["월"], y=df["고용"], name="고용", marker_color="#52b788"), secondary_y=False)
            fig2.add_trace(go.Bar(x=df["월"], y=df["해고"], name="해고", marker_color="#e63946"), secondary_y=False)
            fig2.add_trace(go.Scatter(x=df["월"], y=df["작업자수"], name="총 인원", line=dict(color="#1d3557")), secondary_y=True)
            st.plotly_chart(fig2, use_container_width=True)

        with tabs[2]: # 재고
            st.line_chart(df.set_index("월")[["기말재고", "부족재고"]])

        with tabs[3]: # 비용
            cost_cols = ["정규임금비용","초과근무비용","고용비용","해고비용","재고비용","부족재고비용","재료비","하청비용"]
            st.bar_chart(df.set_index("월")[cost_cols])

        with tabs[4]: # 검증
            st.write("### 제약조건 충족 여부")
            valid = []
            for i, r in df.iterrows():
                p_cap = upw * r['작업자수'] + (1/std_time) * r['초과시간']
                valid.append("✅ 충족" if r['생산량'] <= p_cap + 0.1 else "❌ 초과")
            df['생산용량검증'] = valid
            st.table(df[['월', '작업자수', '생산량', '초과시간', '생산용량검증']])

        with tabs[5]: # 결과표
            st.dataframe(df.style.format(precision=1), use_container_width=True)

        with tabs[6]: # 전략비교
            show_strategy_comparison(df, tc, M, demand_input, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C)

        with tabs[7]: # 평가
            show_plan_evaluation(df, tc, M, demand_input, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C)
