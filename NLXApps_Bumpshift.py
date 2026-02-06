import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import os

# 1. Shift 데이터 전처리 및 분석 함수
def process_shift_data(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    
    # 데이터 로드
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns] 
    
    # [A] 단위 변환 (mm -> um)
    # Bump_Center_X/Y 및 측정된 Shift 값들을 모두 um 단위로 변환합니다.
    df['X'] = df['Bump_Center_X'] * 1000
    df['Y'] = df['Bump_Center_Y'] * 1000
    df['Shift_X_um'] = df['Shift_X'] * 1000
    df['Shift_Y_um'] = df['Shift_Y'] * 1000
    df['Shift_Norm_um'] = df['Shift_Norm'] * 1000
    
    # [B] Shift_Norm 기준 이상치 제거 (IQR 필터링)
    # 전체적인 변위량(Norm)을 기준으로 비정상적인 데이터를 필터링합니다.
    df_clean = df[df['Shift_Norm_um'] != 0].copy()
    q1, q3 = df_clean['Shift_Norm_um'].quantile([0.25, 0.75])
    iqr = q3 - q1
    df_final = df_clean[
        (df_clean['Shift_Norm_um'] >= q1 - 1.5 * iqr) & 
        (df_clean['Shift_Norm_um'] <= q3 + 1.5 * iqr)
    ].copy()

    # [C] Pitch 계산 (기존 코드 로직 유지)
    df_final['Y_grid'] = df_final['Y'].round(0) 
    df_final = df_final.sort_values(by=['Y_grid', 'X'])
    df_final['X_Pitch'] = df_final.groupby('Y_grid')['X'].diff()

    df_final['X_grid'] = df_final['X'].round(0)
    df_final = df_final.sort_values(by=['X_grid', 'Y'])
    df_final['Y_Pitch'] = df_final.groupby('X_grid')['Y'].diff()

    print(f"🧹 데이터 정제 완료 (Shift 이상치 제외)")
    return df_final

# 2. 통계치 계산 및 출력 함수
def print_shift_statistics(df):
    if df is None: return
    
    # 분석 대상 항목
    items = ["Shift_X_um", "Shift_Y_um", "Shift_Norm_um"]
    
    print("="*75)
    print(f"📊 Bump Shift 분석 리포트 (Unit: um)")
    print("-" * 75)
    print(f"{'Item':<15} | {'Average (um)':<15} | {'Std Dev (um)':<15} | {'3-Sigma (um)':<15}")
    print("-" * 75)
    
    for item in items:
        data = df[item].dropna()
        avg = data.mean()
        std_dev = data.std()
        three_sigma = std_dev * 3 if not pd.isna(std_dev) else 0
        
        print(f"{item:<15} | {avg:>15.6f} | {std_dev:>15.6f} | {three_sigma:>15.6f}")
    
    print("-" * 75)
    print(f"✅ Analyzed bumps : {len(df)} units")
    print("="*75)

# 3. 2x2 시각화 함수
def plot_shift_visualizations(df):
    if df is None: return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # [1] Shift Norm Contour Map (전체 변위 분포)
    ax1 = axes[0, 0]
    xi = np.linspace(df['X'].min(), df['X'].max(), 200)
    yi = np.linspace(df['Y'].min(), df['Y'].max(), 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((df['X'], df['Y']), df['Shift_Norm_um'], (xi, yi), method='linear')
    cp = ax1.contourf(xi, yi, zi, cmap='magma', levels=15)
    fig.colorbar(cp, ax=ax1, label='Shift Norm (um)')
    ax1.set_title('Bump Shift Norm Map (um)')

    # [2] Shift X vs Shift Y Scatter Plot (경향성 분석)
    axes[0, 1].axhline(0, color='black', linewidth=1)
    axes[0, 1].axvline(0, color='black', linewidth=1)
    sns.scatterplot(x='Shift_X_um', y='Shift_Y_um', data=df, ax=axes[0, 1], alpha=0.6)
    axes[0, 1].set_title('Shift X vs Shift Y Scatter (um)')
    axes[0, 1].set_xlabel('Shift X (um)')
    axes[0, 1].set_ylabel('Shift Y (um)')

    # [3] Shift X & Y Distribution (Box Plot)
    shift_data = df[['Shift_X_um', 'Shift_Y_um']]
    sns.boxplot(data=shift_data, ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title('Shift X/Y Distribution (um)')

    # [4] Shift Norm Distribution (Histogram)
    sns.histplot(df['Shift_Norm_um'], kde=True, ax=axes[1, 1], color='crimson')
    axes[1, 1].set_title('Shift Norm Frequency (um)')

    plt.tight_layout()
    plt.show()

# --- 메인 실행부 ---
if __name__ == "__main__":
    # 사용자 환경에 맞춰 경로와 파일명을 수정하세요.
    TARGET_FOLDER = 'C:/Users/KSJEOKI1/OneDrive - Carl Zeiss AG/문서/Other Demo/Astar' 
    TARGET_FILE = 'cross_section_shift_raw_data.csv' 

    # 1. 데이터 처리
    shift_data = process_shift_data(TARGET_FOLDER, TARGET_FILE)
    
    if shift_data is not None:
        # 2. 통계 출력
        print_shift_statistics(shift_data)
        # 3. 그래프 출력
        plot_shift_visualizations(shift_data)