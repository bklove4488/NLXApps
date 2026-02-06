import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

# --- [1] 데이터 전처리 및 Pitch 계산 로직 (IQR 필터링 고도화) ---
def process_data(df, scale_factor, apply_iqr, apply_pitch_iqr):
    df.columns = [c.strip() for c in df.columns]
    
    # 데이터 타입 판별
    if 'Height' in df.columns: d_type, target = "Height", "Height"
    elif 'Radius' in df.columns: d_type, target = "Radius", "Radius"
    elif 'Shift_Norm' in df.columns: d_type, target = "Shift", "Shift_Norm"
    else: return None, None

    # 기본 단위 변환
    df['X'] = df['Bump_Center_X'] * scale_factor
    df['Y'] = df['Bump_Center_Y'] * scale_factor
    df['Value'] = df[target] * scale_factor
    
    # 1차: 메인 Value(Height/Radius/Shift) IQR 제거
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    # 2차: Pitch 계산 (그리드 기반)
    df_clean['Y_grid'] = df_clean['Y'].round(0)
    df_clean = df_clean.sort_values(by=['Y_grid', 'X'])
    df_clean['X_Pitch'] = df_clean.groupby('Y_grid')['X'].diff()

    df_clean['X_grid'] = df_clean['X'].round(0)
    df_clean = df_clean.sort_values(by=['X_grid', 'Y'])
    df_clean['Y_Pitch'] = df_clean.groupby('X_grid')['Y'].diff()

    # 3차: [추가] Pitch 데이터 IQR 필터링 (선택 사항)
    if apply_pitch_iqr:
        for col in ['X_Pitch', 'Y_Pitch']:
            p_data = df_clean[col].dropna()
            if not p_data.empty:
                pq1, pq3 = p_data.quantile([0.25, 0.75])
                piqr = pq3 - pq1
                # 이상치에 해당하는 행의 Pitch 값만 NaN으로 처리하여 통계/그래프에서 제외
                df_clean.loc[(df_clean[col] < pq1 - 1.5 * piqr) | (df_clean[col] > pq3 + 1.5 * piqr), col] = np.nan

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Professional Analyzer", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (IQR Advanced)")

# 사이드바: IQR 옵션 세분화
st.sidebar.header("📁 Data & Filtering")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor (mm to um = 1000)", value=1000)

st.sidebar.subheader("🛡️ Outlier Removal Settings")
use_val_iqr = st.sidebar.checkbox("Apply IQR to Value (H/R/S)", value=True)
use_pitch_iqr = st.sidebar.checkbox("Apply IQR to Pitch (X/Y)", value=True) # 추가된 옵션

if uploaded_files:
    all_data = []
    summary_list = []

    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        # 피치 IQR 옵션 전달
        p_df, d_type = process_data(raw_df, scale, use_val_iqr, use_pitch_iqr)
        
        if p_df is not None:
            p_df['Source'] = file.name
            all_data.append(p_df)
            
            v = p_df['Value'].dropna()
            xp = p_df['X_Pitch'].dropna()
            yp = p_df['Y_Pitch'].dropna()
            
            summary_list.append({
                "File": file.name, "Type": d_type, 
                "Avg": v.mean(), "3-Sigma": v.std()*3,
                "X_Pitch Avg": xp.mean(), "X_Pitch 3σ": xp.std()*3,
                "Y_Pitch Avg": yp.mean(), "Y_Pitch 3σ": yp.std()*3,
                "Count": len(v)
            })

    combined_df = pd.concat(all_data)
    
    # 상단 요약 요약 (Pitch IQR 반영됨)
    st.subheader("📊 Statistics Summary (IQR Applied)")
    st.dataframe(pd.DataFrame(summary_list).style.highlight_min(axis=0, subset=['3-Sigma', 'X_Pitch 3σ']))

    # 상세 분석 대상 선택
    target_file = st.selectbox("Select File for Detail View", [f.name for f in uploaded_files])
    plot_df = combined_df[combined_df['Source'] == target_file]

    # 상세 수치 테이블
    st.markdown("---")
    st.write(f"### 🔢 Detailed Numerical Report: {target_file}")
    col_stat1, col_stat2 = st.columns([1, 2])
    with col_stat1:
        st.write("**Pitch Statistics (Filtered)**")
        p_stats = plot_df[['X_Pitch', 'Y_Pitch']].describe().loc[['mean', 'std']]
        p_stats.loc['3-Sigma'] = p_stats.loc['std'] * 3
        st.table(p_stats)

    # --- [3] 시각화 커스터마이징 및 실행 ---
    st.subheader("🎨 Plot Settings")
    c1, c2, c3, c4 = st.columns(4)
    plots_meta = {
        "Contour": {"title": f"{d_type} Map", "xl": "X (um)", "yl": "Y (um)"},
        "Histogram": {"title": f"{d_type} Dist", "xl": "Value", "yl": "Freq"},
        "Pitch": {"title": "Pitch Spread (IQR Applied)", "xl": "Axis", "yl": "Pitch (um)"},
        "Boxplot": {"title": "Total Comparison", "xl": "File", "yl": "Value"}
    }
    
    config = {}
    for i, (k, v) in enumerate(plots_meta.items()):
        with [c1, c2, c3, c4][i]:
            t = st.text_input(f"Title ({k})", v['title'])
            xl = st.text_input(f"X ({k})", v['xl'])
            yl = st.text_input(f"Y ({k})", v['yl'])
            m_sc = st.checkbox(f"Manual Scale ({k})")
            y_lim = None
            if m_sc:
                # 피치 그래프일 경우 피치 데이터 기준으로 초기값 설정
                ref_data = plot_df['X_Pitch'] if k == "Pitch" else plot_df['Value']
                y_min = st.number_input(f"Min_{k}", value=0.0)
                y_max = st.number_input(f"Max_{k}", value=float(ref_data.max()))
                y_lim = (y_min, y_max)
            config[k] = {"t": t, "xl": xl, "yl": yl, "ylim": y_lim}

    # 그래프 그리기
    st.markdown("---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Contour
    ax1 = axes[0, 0]
    xi = np.linspace(plot_df['X'].min(), plot_df['X'].max(), 100)
    yi = np.linspace(plot_df['Y'].min(), plot_df['Y'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((plot_df['X'], plot_df['Y']), plot_df['Value'], (xi, yi), method='linear')
    cp = ax1.contourf(xi, yi, zi, cmap='viridis', levels=15)
    plt.colorbar(cp, ax=ax1)
    ax1.set_title(config["Contour"]["t"]); ax1.set_xlabel(config["Contour"]["xl"]); ax1.set_ylabel(config["Contour"]["yl"])
    if config["Contour"]["ylim"]: ax1.set_ylim(config["Contour"]["ylim"])

    # 2. Histogram
    ax2 = axes[0, 1]
    sns.histplot(plot_df['Value'], kde=True, ax=ax2, color='skyblue')
    ax2.set_title(config["Histogram"]["t"]); ax2.set_xlabel(config["Histogram"]["xl"]); ax2.set_ylabel(config["Histogram"]["yl"])
    if config["Histogram"]["ylim"]: ax2.set_xlim(config["Histogram"]["ylim"])

    # 3. Pitch Boxplot (IQR 반영된 데이터 사용)
    ax3 = axes[1, 0]
    pitch_melt = plot_df[['X_Pitch', 'Y_Pitch']].melt(var_name='Type', value_name='Pitch')
    sns.boxplot(x='Type', y='Pitch', data=pitch_melt, ax=ax3, palette='Set2')
    ax3.set_title(config["Pitch"]["t"]); ax3.set_xlabel(config["Pitch"]["xl"]); ax3.set_ylabel(config["Pitch"]["yl"])
    if config["Pitch"]["ylim"]: ax3.set_ylim(config["Pitch"]["ylim"])

    # 4. Global Boxplot
    ax4 = axes[1, 1]
    sns.boxplot(x='Source', y='Value', data=combined_df, ax=ax4)
    ax4.set_title(config["Boxplot"]["t"]); ax4.set_xlabel(config["Boxplot"]["xl"]); ax4.set_ylabel(config["Boxplot"]["yl"])
    if config["Boxplot"]["ylim"]: ax4.set_ylim(config["Boxplot"]["ylim"])

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("💡 사이드바에서 CSV 파일들을 업로드하세요.")