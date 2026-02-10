import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.cluster import KMeans  # 레이어 분리를 위해 추가

# --- [1] 데이터 전처리 및 레이어 분석 로직 ---
def process_data(df, scale_factor, apply_iqr, apply_pitch_iqr):
    df.columns = [c.strip() for c in df.columns]
    
    # 데이터 타입 판별
    if 'Height' in df.columns: d_type, target = "Height", "Height"
    elif 'Radius' in df.columns: d_type, target = "Radius", "Radius"
    elif 'Shift_Norm' in df.columns: d_type, target = "Shift", "Shift_Norm"
    else: return None, None

    # [추가] 레이어 자동 분석 (Z-Position 기반 클러스터링)
    # Z값의 차이가 미세하므로 클러스터링을 통해 층을 구분합니다.
    z_values = df['Bump_Center_Z'].values.reshape(-1, 1)
    
    # 엘보우 포인트 대신 최대 5개 층까지 탐색하여 최적의 층 수 계산 (간단한 로직)
    # 실무적으로는 사용자가 층 수를 입력하게 할 수도 있습니다.
    n_clusters = 1
    if len(df) > 10:
        # Z값의 고유값 범위를 보고 대략적인 층수 추정 (차이가 0.005 이상일 때 구분 등)
        z_range = np.ptp(df['Bump_Center_Z'])
        if z_range > 0.01: n_clusters = 2 # 예시 임계치
        if z_range > 0.05: n_clusters = 3
    
    # 사이드바에서 선택할 수 있도록 일단 1~5층 사이에서 자동 할당하거나 
    # 아래 메인 루프에서 사용자가 지정한 n_layers를 사용할 수 있습니다.
    
    # 기본 단위 변환
    df['X'] = df['Bump_Center_X'] * scale_factor
    df['Y'] = df['Bump_Center_Y'] * scale_factor
    df['Z_um'] = df['Bump_Center_Z'] * scale_factor
    df['Value'] = df[target] * scale_factor
    
    # 1차: 메인 Value IQR 제거
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    # 2차: Pitch 계산
    df_clean['Y_grid'] = df_clean['Y'].round(0)
    df_clean = df_clean.sort_values(by=['Y_grid', 'X'])
    df_clean['X_Pitch'] = df_clean.groupby('Y_grid')['X'].diff()

    df_clean['X_grid'] = df_clean['X'].round(0)
    df_clean = df_clean.sort_values(by=['X_grid', 'Y'])
    df_clean['Y_Pitch'] = df_clean.groupby('X_grid')['Y'].diff()

    # 3차: Pitch IQR 필터링
    if apply_pitch_iqr:
        for col in ['X_Pitch', 'Y_Pitch']:
            p_data = df_clean[col].dropna()
            if not p_data.empty:
                pq1, pq3 = p_data.quantile([0.25, 0.75])
                piqr = pq3 - pq1
                df_clean.loc[(df_clean[col] < pq1 - 1.5 * piqr) | (df_clean[col] > pq3 + 1.5 * piqr), col] = np.nan

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Multi-Layer Analyzer", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (Layer Analysis)")

st.sidebar.header("📁 Data & Layer Settings")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)

# [추가] 레이어 분리 설정
n_layers = st.sidebar.slider("Number of expected layers (Z-axis)", 1, 5, 1)

st.sidebar.subheader("🛡️ Outlier Removal Settings")
use_val_iqr = st.sidebar.checkbox("Apply IQR to Value", value=True)
use_pitch_iqr = st.sidebar.checkbox("Apply IQR to Pitch", value=True)

if uploaded_files:
    all_data = []
    
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, use_val_iqr, use_pitch_iqr)
        
        if p_df is not None:
            # Z축 클러스터링 수행 (레이어 할당)
            if n_layers > 1:
                kmeans = KMeans(n_clusters=n_layers, random_state=42)
                p_df['Layer'] = kmeans.fit_predict(p_df[['Bump_Center_Z']])
                # Z값 평균 순서대로 레이어 이름 재정렬 (0층이 가장 낮은 층이 되도록)
                layer_order = p_df.groupby('Layer')['Bump_Center_Z'].mean().sort_values().index
                layer_map = {old: new for new, old in enumerate(layer_order)}
                p_df['Layer'] = p_df['Layer'].map(layer_map)
            else:
                p_df['Layer'] = 0
                
            p_df['Source'] = file.name
            all_data.append(p_df)

    combined_df = pd.concat(all_data)

    # 레이어 필터링 UI
    st.sidebar.markdown("---")
    unique_layers = sorted(combined_df['Layer'].unique())
    selected_layer = st.sidebar.selectbox("Select Layer to View", ["All Layers"] + [f"Layer {i}" for i in unique_layers])

    # 데이터 필터링 실행
    if selected_layer != "All Layers":
        layer_num = int(selected_layer.split(" ")[1])
        display_df = combined_df[combined_df['Layer'] == layer_num]
    else:
        display_df = combined_df

    # 상단 요약 요약
    st.subheader(f"📊 Statistics Summary ({selected_layer})")
    summary_list = []
    for src in display_df['Source'].unique():
        sub = display_df[display_df['Source'] == src]
        summary_list.append({
            "File": src, "Avg": sub['Value'].mean(), "3-Sigma": sub['Value'].std()*3,
            "Count": len(sub)
        })
    st.dataframe(pd.DataFrame(summary_list))

    # [이후 시각화 로직은 display_df를 사용하여 기존과 동일하게 진행...]
    # (생략: 기존 코드의 시각화 부분에서 plot_df를 display_df 기반으로 필터링하여 사용)