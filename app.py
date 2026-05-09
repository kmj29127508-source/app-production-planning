import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. 헬퍼 함수 정의 (함수 호출 전 상단 배치)
# ─────────────────────────────────────────────

def show_strategy_comparison(df, tc, M, demand_list, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C):
    st.markdown('<div class="sec">🆚 전략별 비용 비교 분석</div>', unsafe_allow_html=True)
    chase_cost = sum(demand_list) * (c_P + c_W/M['upw']) 
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
    st.info(f"**전문가 권고:** 현재 가동률은 {avg_util:.1f}%이며, 외주 비용은 {df['하청비용'].sum():,.0f}원 발생 중입니다.")

# ─────────────────────────────────────────────
# 2. Pyomo 최적화 엔진 (오류 수정 핵심)
# ─────────────────────────────────────────────
try:
    from pyomo.environ import (
        ConcreteModel, Var, Objective, Constraint, ConstraintList,
        NonNegativeReals, NonNegativeIntegers, SolverFactory, minimize, value
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
    
    # 모든 변수를 TIME(0~TH) 범위로 정의하여 인덱스 에러 방지
    m.W = Var(TIME, domain=tv, bounds=(0, None))
    m.H = Var(TIME, domain=tv, bounds=(0, None))
    m.L = Var(TIME, domain=tv, bounds=(0, None))
    m.P = Var(TIME, domain=tv, bounds=(0, None))
    m.I = Var(TIME, domain=tv, bounds=(0, None))
    m.S = Var(TIME, domain=tv, bounds=(0, None))
    m.C = Var(TIME, domain=tv, bounds=(0, None))
    m.O = Var(TIME, domain=tv, bounds=(0, None))

    # 목적함수
    m.Cost = Objective(expr=sum(
        c_W*m.W[t] + c_O*m.O[t] + c_H*m.H[t] + c_L*m.L[t] +
        c_I*m.I[t] + c_S*m.S[t] + c_P*m.P[t] + c_C*m.C[t] for t in T
    ), sense=minimize)

    # 제약조건 (AttributeError: Constraint.List 해결 -> ConstraintList)
    m.cons = ConstraintList()
    for t in T:
        m.cons.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        m.cons.add(m.P[t] <= upw * m.W[t] + (1.0/std_time) * m.O[t])
        m.cons.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - demand[t-1] - m.S[t-1] + m.S[t])
        m.cons.add(m.O[t] <= ot_limit * m.W[t])

    # 초기/기말 제약
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

    if not solved: return None, "사용 가능한 솔버(glpk/cbc/highs)가 없습니다."

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
# 3. Streamlit UI (기존 레이아웃 유지)
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
.sec{ font-size:1rem; font-weight:700; color:#1a3c2e; border-left:4px solid #2d7a4f; padding-left:.7rem; margin:1.1rem 0 .
