import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import os

# 1. 데이터 전처리 및 Pitch 계산 함수
def process_bump_data(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns] 
    
    # [A] 단위 변환 (mm -> um)
    df['X'] = df['Bump_Center_X'] * 1000
    df['Y'] = df['Bump_Center_Y'] * 1000
    df['Height'] = df['Height'] * 1000
    
    # [B] Height 이상치 제거 (IQR 필터링)
    df_clean = df[df['Height'] != 0].copy()
    qh1, qh3 = df_clean['Height'].quantile([0.25, 0.75])
    iqr_h = qh3 - qh1
    df_final = df_clean[
        (df_clean['Height'] >= qh1 - 1.5 * iqr_h) & 
        (df_clean['Height'] <= qh3 + 1.5 * iqr_h)
    ].copy()

    # [C] 양산형 Pitch 계산 (좌표 그리드 기반)
    df_final['Y_grid'] = df_final['Y'].round(0) 
    df_final = df_final.sort_values(by=['Y_grid', 'X'])
    df_final['X_Pitch'] = df_final.groupby('Y_grid')['X'].diff()

    df_final['X_grid'] = df_final['X'].round(0)
    df_final = df_final.sort_values(by=['X_grid', 'Y'])
    df_final['Y_Pitch'] = df_final.groupby('X_grid')['Y'].diff()

    # [D] Pitch 이상치 제거 (IQR 필터링)
    for col in ['X_Pitch', 'Y_Pitch']:
        valid_p = df_final[col].dropna()
        if not valid_p.empty:
            q1, q3 = valid_p.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df_final.loc[(df_final[col] < lower) | (df_final[col] > upper), col] = np.nan

    print(f"🧹 데이터 정제 완료 (Height/Pitch 이상치 제외)")
    return df_final

# 2. 통계치 계산 및 출력 함수 (Average, Std Dev, 3-Sigma 추가)
def print_statistics(df):
    if df is None: return
    
    items = ["Height", "X_Pitch", "Y_Pitch"]
    
    print("="*75)
    print(f"📊 Bump 데이터 분석 리포트 (Unit: um)")
    print("-" * 75)
    print(f"{'Item':<12} | {'Average (um)':<15} | {'Std Dev (um)':<15} | {'3-Sigma (um)':<15}")
    print("-" * 75)
    
    for item in items:
        data = df[item].dropna()
        avg = data.mean()
        std_dev = data.std()
        three_sigma = std_dev * 3 if not pd.isna(std_dev) else 0
        
        print(f"{item:<12} | {avg:>15.6f} | {std_dev:>15.6f} | {three_sigma:>15.6f}")
    
    print("-" * 75)
    print(f"✅ Analyzed bumps : {len(df)} units")
    print("="*75)

# 3. 2x2 시각화 함수
def plot_visualizations(df):
    if df is None: return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # [1] Height Contour Map
    ax1 = axes[0, 0]
    xi = np.linspace(df['X'].min(), df['X'].max(), 200)
    yi = np.linspace(df['Y'].min(), df['Y'].max(), 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((df['X'], df['Y']), df['Height'], (xi, yi), method='linear')
    cp = ax1.contourf(xi, yi, zi, cmap='viridis', levels=15)
    fig.colorbar(cp, ax=ax1, label='Height (um)')
    ax1.set_title('Bump Height Map')

    # [2] Height Box Plot
    sns.boxplot(y=df['Height'], ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_title('Height Distribution (um)')

    # [3] X-Pitch Box Plot
    sns.boxplot(y=df['X_Pitch'].dropna(), ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('X-Pitch Distribution (um)')

    # [4] Y-Pitch Box Plot
    sns.boxplot(y=df['Y_Pitch'].dropna(), ax=axes[1, 1], color='salmon')
    axes[1, 1].set_title('Y-Pitch Distribution (um)')

    plt.tight_layout()
    plt.show()

# --- 메인 실행부 ---
if __name__ == "__main__":
    TARGET_FOLDER = 'C:/Users/KSJEOKI1/OneDrive - Carl Zeiss AG/문서/Other Demo/Astar' 
    TARGET_FILE = 'Astar_bump_height.csv' 

    bump_data = process_bump_data(TARGET_FOLDER, TARGET_FILE)
    
    if bump_data is not None:
        print_statistics(bump_data)
        plot_visualizations(bump_data)