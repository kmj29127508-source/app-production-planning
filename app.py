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
# 1. Pyomo 및 솔버 설정
# ─────────────────────────────────────────────
try:
    from pyomo.environ import (
        ConcreteModel, Var, Objective, Constraint, ConstraintList,
        NonNegativeReals, NonNegativeIntegers, SolverFactory, minimize, value
    )
    PYOMO_OK = True
except ImportError:
    PYOMO_OK = False

# ─────────────────────────────────────────────
# 2. 헬퍼 함수 (전략 계산 및 시각화) - 호출 전 정의
# ─────────────────────────────────────────────

def calc_strategy(demand, W0, I0, I_final, strategy, upw, std_time, ot_limit,
                  c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C):
    """단일 전략 수동 계산 (비교용 휴리스틱)"""
    TH = len(demand)
    results = []
    inv = I0
    workforce = W0
    total_cost = 0

    for t in range(TH):
        d = demand[t]
        if strategy == "Level Production (평준화)":
            avg_d = sum(demand) / TH
            prod = min(upw * W0, avg_d) 
            overtime = 0
        elif strategy == "Chase Demand (추종)":
            needed_workers = max(1, int(np.ceil(d / upw)))
            workforce = needed_workers
            prod = upw * workforce
            overtime = 0
        elif strategy == "Overtime-Only (초과근무)":
            workforce = W0
            base = upw * workforce
            extra_needed = max(0, d - base - inv)
            overtime = min(extra_needed * std_time, ot_limit * workforce)
            prod = base + overtime / std_time
        else: prod, overtime = 0, 0

        end_inv = inv + prod - d
        backorder = 0
        if end_inv < 0:
            backorder = -end_inv
            end_inv = 0

        prev_w = W0 if t == 0 else results[-1]["작업자수"]
        hire = max(0, workforce - prev_w)
        fire = max(0, prev_w - workforce)

        cost = (c_W*workforce + c_O*overtime + c_H*hire + c_L*fire
                + c_I*end_inv + c_S*backorder + c_P*prod)
        total_cost += cost

        results.append({
            "월": f"{t+1}월", "수요": d, "작업자수": round(workforce, 1),
            "생산량": round(prod, 1), "기말재고": round(end_inv, 1),
            "부족재고": round(backorder, 1), "초과시간": round(overtime, 1),
            "외주량": 0, "월비용": round(cost, 1), "고용": hire, "해고": fire
        })
        inv = end_inv
    return pd.DataFrame(results), round(total_cost, 1), max(0, (1 - sum(r['부족재고'] for r in results)/sum(demand))*100)

def show_strategy_comparison(df_opt, tc_opt, M, demand_list, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C):
    st.markdown('<div class="sec">📊 4가지 생산 전략 비교</div>', unsafe_allow_html=True)
    strategies = ["Level Production (평준화)", "Chase Demand (추종)", "Overtime-Only (초과근무)"]
    
    res_sum = [{
        "전략": "Pyomo LP (최적화)", "총비용": tc_opt,
        "총생산": round(df_opt["생산량"].sum() + df_opt["외주량"].sum(), 1),
        "평균인력": round(df_opt["작업자수"].mean(), 1),
        "총초과시간": round(df_opt["초과시간"].sum(), 1),
        "총재고": round(df_opt["기말재고"].sum(), 1),
        "서비스레벨%": round((1 - df_opt["부족재고"].sum()/sum(demand_list))*100, 1)
    }]

    for sn in strategies:
        df_s, tc_s, svc_s = calc_strategy(demand_list, M["W0"], M["I0"], M["I_final"], sn, M["upw"], M["std_time"], M["ot_limit"], c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C)
        res_sum.append({
            "전략": sn, "총비용": tc_s, "총생산": df_s["생산량"].sum(),
            "평균인력": df_s["작업자수"].mean(), "총초과시간": df_s["초과시간"].sum(),
            "총재고": df_s["기말재고"].sum(), "서비스레벨%": svc_s
        })

    df_comp = pd.DataFrame(res_sum)
    st.dataframe(df_comp.style.highlight_min(subset=['총비용'], color='#d1e7dd').format(precision=1), use_container_width=True, hide_index=True)
    
    fig = go.Figure()
    for i, row in df_comp.iterrows():
        fig.add_trace(go.Bar(x=[row["전략"]], y=[row["총비용"]], name=row["전략"], text=f"{row['총비용']:,.0f}", textposition='outside'))
    fig.update_layout(height=400, title="전략별 총비용 비교", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_plan_evaluation(df_opt, tc_opt, M, demand_list, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C):
    st.markdown('<div class="sec">💡 계획 평가 및 권고</div>', unsafe_allow_html=True)
    total_backlog = df_opt["부족재고"].sum()
    if total_backlog > 0:
        st.warning(f"⚠️ 현재 계획에서 총 {total_backlog:,.0f} 단위의 부족재고가 발생합니다.")
    else:
        st.success("✅ 수요를 완벽히 충족하는 최적 계획입니다.")
    
    k1, k2 = st.columns(2)
    k1.metric("최대 재고 수준", f"{df_opt['기말재고'].max():,.0f}")
    k2.metric("인력 변동(고용+해고)", f"{df_opt['고용'].sum() + df_opt['해고'].sum():,.0f}명")

# ─────────────────────────────────────────────
# 3. 최적화 엔진 (Pyomo)
# ─────────────────────────────────────────────
def solve_app(demand, W0, I0, I_final, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C, ot_limit, upw, std_time, model_type="LP"):
    if not PYOMO_OK: return None, "Pyomo 설치 필요"
    TH = len(demand); T = range(1, TH + 1); TIME = range(0, TH + 1)
    m = ConcreteModel()
    tv = NonNegativeIntegers if model_type == "IP" else NonNegativeReals
    
    m.W = Var(TIME, domain=tv, bounds=(0, None))
    m.H = Var(TIME, domain=tv, bounds=(0, None))
    m.L = Var(TIME, domain=tv, bounds=(0, None))
    m.P = Var(TIME, domain=tv, bounds=(0, None))
    m.I = Var(TIME, domain=tv, bounds=(0, None))
    m.S = Var(TIME, domain=tv, bounds=(0, None))
    m.C = Var(TIME, domain=tv, bounds=(0, None))
    m.O = Var(TIME, domain=tv, bounds=(0, None))

    m.obj = Objective(expr=sum(c_W*m.W[t] + c_O*m.O[t] + c_H*m.H[t] + c_L*m.L[t] + c_I*m.I[t] + c_S*m.S[t] + c_P*m.P[t] + c_C*m.C[t] for t in T), sense=minimize)
    m.cons = ConstraintList()
    for t in T:
        m.cons.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        m.cons.add(m.P[t] <= upw * m.W[t] + (1.0/std_time) * m.O[t])
        m.cons.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - demand[t-1] - m.S[t-1] + m.S[t])
        m.cons.add(m.O[t] <= ot_limit * m.W[t])
    m.cons.add(m.W[0] == W0); m.cons.add(m.I[0] == I0); m.cons.add(m.S[0] == 0)
    m.cons.add(m.I[TH] >= I_final); m.cons.add(m.S[TH] == 0)

    for sn in ["glpk", "cbc", "highs"]:
        try:
            slv = SolverFactory(sn)
            if slv.available():
                slv.solve(m)
                rows = []
                for t in T:
                    rows.append({
                        "월": f"{t}월", "수요": demand[t-1], "작업자수": value(m.W[t]), "고용": value(m.H[t]), "해고": value(m.L[t]),
                        "생산량": value(m.P[t]), "기말재고": value(m.I[t]), "부족재고": value(m.S[t]), "외주량": value(m.C[t]), "초과시간": value(m.O[t]),
                        "정규임금비용": c_W*value(m.W[t]), "초과근무비용": c_O*value(m.O[t]), "고용비용": c_H*value(m.H[t]), "해고비용": c_L*value(m.L[t]),
                        "재고비용": c_I*value(m.I[t]), "부족재고비용": c_S*value(m.S[t]), "재료비": c_P*value(m.P[t]), "하청비용": c_C*value(m.C[t])
                    })
                df = pd.DataFrame(rows)
                df["총비용"] = df.iloc[:, 10:].sum(axis=1)
                return df, value(m.obj)
        except: continue
    return None, "솔버 에러"

# ─────────────────────────────────────────────
# 4. 메인 UI (Streamlit)
# ─────────────────────────────────────────────
st.set_page_config(page_title="원예장비 APP", page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
html,body,[class*="css"]{ font-family:'Noto Sans KR',sans-serif; }
.hdr{ background:linear-gradient(135deg,#1a4731,#2d7a4f); padding:1.5rem; border-radius:12px; color:white; margin-bottom:1.5rem; }
.kpi{ background:#f0f7f4; border-left:5px solid #2d7a4f; border-radius:10px; padding:1rem; text-align:center; }
.kpi-lbl{ font-size:0.8rem; color:#5a7a6a; font-weight:600; }
.kpi-val{ font-size:1.4rem; font-weight:700; color:#1a3c2e; }
.sec{ font-size:1.1rem; font-weight:700; color:#1a3c2e; border-left:4px solid #2d7a4f; padding-left:0.8rem; margin:1.5rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hdr"><h1>🌿 원예장비 제조업체 총괄생산계획 (APP)</h1><p>스마트제조_06 강의록 기반 최적화 시스템</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 입력 파라미터")
    n = st.selectbox("계획 기간", [6, 12], index=0)
    demand_list = []
    preset = [1600, 3000, 3200, 3800, 2200, 2200] + [2500]*6
    c1, c2 = st.columns(2)
    for i in range(n):
        with c1 if i%2==0 else c2:
            v = st.number_input(f"{i+1}월 수요", value=preset[i], key=f"d{i}")
            demand_list.append(v)
    
    W0 = st.number_input("초기 인원", value=80)
    I0 = st.number_input("초기 재고", value=1000)
    I_final = st.number_input("목표 기말재고", value=500)
    
    with st.expander("💰 비용 계수"):
        c_W = st.number_input("정규임금", value=640.0); c_O = st.number_input("초과임금", value=6.0)
        c_H = st.number_input("고용비용", value=300.0); c_L = st.number_input("해고비용", value=500.0)
        c_I = st.number_input("재고비용", value=2.0); c_S = st.number_input("부족비용", value=5.0)
        c_P = st.number_input("재료비", value=10.0); c_C = st.number_input("외주비용", value=30.0)

    ot_limit = st.number_input("초과시간 한도", value=10)
    std_time = st.number_input("표준시간", value=4.0)
    upw = (20 * 8) / std_time
    mt = st.radio("모델", ["LP", "IP"])
    run = st.button("🚀 최적화 실행", type="primary", use_container_width=True)

if run or 'df' in st.session_state:
    if run:
        df, tc = solve_app(demand_list, W0, I0, I_final, c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C, ot_limit, upw, std_time, mt)
        st.session_state.df, st.session_state.tc = df, tc
        st.session_state.meta = {"W0":W0, "I0":I0, "I_final":I_final, "upw":upw, "std_time":std_time, "ot_limit":ot_limit, "n":n, "mt":mt, "demand_list":demand_list, "c_W":c_W, "c_O":c_O, "c_H":c_H, "c_L":c_L, "c_I":c_I, "c_S":c_S, "c_P":c_P, "c_C":c_C}

    if st.session_state.get('df') is not None:
        df, tc, M = st.session_state.df, st.session_state.tc, st.session_state.meta
        
        # KPI 대시보드
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi"><div class="kpi-lbl">총 비용</div><div class="kpi-val">{tc:,.0f}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi"><div class="kpi-lbl">평균 재고</div><div class="kpi-val">{df["기말재고"].mean():,.0f}</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi"><div class="kpi-lbl">총 외주량</div><div class="kpi-val">{df["외주량"].sum():,.0f}</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi"><div class="kpi-lbl">서비스율</div><div class="kpi-val">{(1-df["부족재고"].sum()/sum(demand_list))*100:.1f}%</div></div>', unsafe_allow_html=True)

        tabs = st.tabs(["🏭 생산개요", "👷 인력계획", "📦 재고분석", "💰 비용분석", "🔍 제약검증", "📋 상세 결과표", "🆚 전략 비교", "💡 계획 평가"])
        
        with tabs[0]: # 생산개요
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["월"], y=df["생산량"], name="정규생산", marker_color="#2d7a4f"))
            fig.add_trace(go.Bar(x=df["월"], y=df["외주량"], name="외주", marker_color="#e76f51"))
            fig.add_trace(go.Scatter(x=df["월"], y=df["수요"], name="수요", line=dict(color="black", dash="dash")))
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[4]: # 제약검증
            df['Capacity_OK'] = df.apply(lambda r: "✅" if r['생산량'] <= (M['upw']*r['작업자수'] + r['초과시간']/M['std_time']) + 0.1 else "❌", axis=1)
            st.table(df[['월', '작업자수', '생산량', '초과시간', 'Capacity_OK']])

        with tabs[5]: # 상세 결과표 (두 번째 코드 내용 적용)
            st.markdown('<div class="sec">📋 월별 총괄생산계획 상세 결과</div>', unsafe_allow_html=True)
            st.code(f"모델: {M['mt']} | 계획기간: {M['n']}개월 | 최소총비용: {tc:,.1f} 천원")
            disp = df.copy()
            sr = {"월": "합계/평균"}
            for col in disp.columns[1:]:
                sr[col] = round(disp[col].mean(), 1) if col=="작업자수" else round(disp[col].sum(), 1)
            disp = pd.concat([disp, pd.DataFrame([sr])], ignore_index=True)
            st.dataframe(disp.style.apply(lambda s: ["font-weight:bold;background:#e8f5ee"]*len(s) if s.name==len(disp)-1 else [""]*len(s), axis=1).format(precision=1), use_container_width=True, hide_index=True)
            st.download_button("📥 CSV 다운로드", disp.to_csv(index=False, encoding="utf-8-sig"), "APP_Result.csv", "text/csv")

        with tabs[6]: # 전략 비교
            show_strategy_comparison(df, tc, M, M['demand_list'], M['c_W'], M['c_O'], M['c_H'], M['c_L'], M['c_I'], M['c_S'], M['c_P'], M['c_C'])

        with tabs[7]: # 계획 평가
            show_plan_evaluation(df, tc, M, M['demand_list'], M['c_W'], M['c_O'], M['c_H'], M['c_L'], M['c_I'], M['c_S'], M['c_P'], M['c_C'])

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:.78rem;">🌿 원예장비 제조업체 APP 시스템 · Hongik University</div>', unsafe_allow_html=True)
