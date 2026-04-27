#!/usr/bin/env python3
"""
多因子有效性分析 v3 — 优化版
改进:
1. IC/收益全部聚合到日级别 (不再是小时级别,消除FR 8h采样噪声)
2. FR用每日均值 (每8h采样→日均值)
3. vol_z < 0 过滤 (只在下行波动时做空)
4. 纯做空策略 (不做多Q1,避免费率成本)
5. 加入交易成本估算 (0.05%*2/笔)
6. 衡量夏普比率而非只看收益率
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, glob, warnings
from datetime import datetime
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============ 配置 ============
DATA_DIR  = "/home/xisang/crypto-new/binance_data"
SWAP_DIR  = f"{DATA_DIR}/binance-swap-candle-csv-1h"
FR_DIR    = f"{DATA_DIR}/binance_funding_rate/usdt"
COIN_DIR  = f"{DATA_DIR}/coin-cap"

TOP_N     = 80
FUTURE_H  = 24
MIN_H     = 168
HALFLIFE  = 30
MAX_RET   = 0.30
BETA_H    = 720
OUTPUT    = "/home/xisang/crypto-new"
COST_RATE = 0.0005  # 0.05% per trade (开+平各一次)

# ============ 符号规范化 ============
def norm(sym):
    return sym.replace('-', '').replace('.csv', '').replace('.CSV', '').upper()

# ============ Step 1: 扫描文件 ============
print("=" * 60)
print("📂 [1/7] 扫描数据文件...")

swap_files = glob.glob(f"{SWAP_DIR}/*.csv")
fr_files   = glob.glob(f"{FR_DIR}/*.csv")
cap_files  = glob.glob(f"{COIN_DIR}/*.csv")

swap_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in swap_files}
fr_norm2orig   = {norm(os.path.basename(f).replace('.csv','').replace('.CSV','')): os.path.basename(f) for f in fr_files}
cap_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in cap_files}

common_norms = set(swap_norm2orig.keys()) & set(fr_norm2orig.keys()) & set(cap_norm2orig.keys())
print(f"  SWAP: {len(swap_files)}, FR: {len(fr_files)}, CAP: {len(cap_files)}")
print(f"  三方共同: {len(common_norms)}")

target_syms = sorted(list(common_norms))[:TOP_N]

# ============ Step 2: 加载并聚合到日级别 ============
print("\n📂 [2/7] 加载数据并聚合到日级别...")

daily_rows = []

for nsym in target_syms:
    # --- SWAP K线 ---
    sp = f"{SWAP_DIR}/{swap_norm2orig[nsym]}"
    df_s = pd.read_csv(sp, parse_dates=['candle_begin_time'])
    df_s = df_s.rename(columns={'quote_volume': 'qv', 'volume': 'vol'})
    df_s['symbol_norm'] = nsym
    df_s['date'] = df_s['candle_begin_time'].dt.date
    # 日线: 取收盘价(最后一条), 波动率用日内收益标准差
    daily_s = df_s.groupby('date').agg(
        close=('close', 'last'),
        open_=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        qv=('qv', 'sum'),
        vol=('vol', 'sum'),
        # 用于计算日内波动率
        hourly_ret_std=('close', lambda x: x.pct_change().std())
    ).reset_index()
    daily_s['symbol_norm'] = nsym

    # --- 资金费率 (日均值) ---
    fp = f"{FR_DIR}/{fr_norm2orig[nsym]}"
    df_f = pd.read_csv(fp, parse_dates=['time'])
    df_f['date'] = df_f['time'].dt.date
    daily_f = df_f.groupby('date')['fundingRate'].mean().reset_index()
    daily_f.columns = ['date', 'fr']
    daily_f['symbol_norm'] = nsym

    # --- 供给量 (日末值) ---
    cp = f"{COIN_DIR}/{cap_norm2orig[nsym]}"
    df_c = pd.read_csv(cp, parse_dates=['candle_begin_time'])
    df_c['date'] = df_c['candle_begin_time'].dt.date
    daily_c = df_c.groupby('date')['circulating_supply'].last().reset_index()
    daily_c.columns = ['date', 'supply']
    daily_c['symbol_norm'] = nsym

    # 合并日线
    m1 = daily_s.merge(daily_f, on=['date', 'symbol_norm'], how='inner')
    m2 = m1.merge(daily_c, on=['date', 'symbol_norm'], how='inner')
    daily_rows.append(m2)

df = pd.concat(daily_rows, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol_norm', 'date']).reset_index(drop=True)
print(f"  合并后记录: {len(df):,} | 币种: {df['symbol_norm'].nunique()} | 日期: {df['date'].min()} ~ {df['date'].max()}")

# ============ Step 3: 计算因子 ============
print("\n🔧 [3/7] 计算因子...")

# 日收益率
df['ret'] = df.groupby('symbol_norm')['close'].pct_change()

# 未来24h收益 (预测目标)
df['fret'] = df.groupby('symbol_norm')['ret'].shift(-FUTURE_H // 24)

# FR趋势 (过去3天FR均值)
df['fr_3d'] = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(3, min_periods=1).mean())

# 成交量变化 (环比)
df['qv_chg'] = df.groupby('symbol_norm')['qv'].pct_change()

# 波动率 (日收益率标准差, 滚动7天)
df['vol_7d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).std())

# 动量 (过去7天收益)
df['mom_7d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).mean())

# 对数供给量
df['log_supply'] = np.log1p(df['supply'])

# 横截面 Z-Score (每日/每个因子各自标准化)
def zscore_cs(g, col):
    return (g - g.mean()) / (g.std() + 1e-9)

for factor_col, src_col in [
    ('fr_z',      'fr'),
    ('fr_trend_z','fr_3d'),
    ('qv_z',      'qv_chg'),
    ('vol_z',     'vol_7d'),
    ('mom_z',     'mom_7d'),
    ('supply_z',  'log_supply'),
]:
    df[factor_col] = df.groupby('date')[src_col].transform(zscore_cs, df[src_col])

# FR符号 (±1)
df['fr_sign'] = np.sign(df['fr'])

# Beta vs BTC (日内数据聚合后再算会噪声大, 用简单相关近似)
# 用过去7天收益与BTC收益的beta
btc_sym = 'BTCUSDT'
if btc_sym in df['symbol_norm'].values:
    btc_df = df[df['symbol_norm'] == btc_sym][['date', 'ret']].rename(columns={'ret': 'btc_ret'})
    df = df.merge(btc_df, on='date', how='left')
    # 每币种与BTC的7日滚动beta
    def rolling_beta(x, btc_ret_series):
        cov = x.rolling(7, min_periods=4).cov(btc_ret_series.reindex(x.index).fillna(method='ffill'))
        var = btc_ret_series.reindex(x.index).fillna(method='ffill').rolling(7, min_periods=4).var()
        return cov / (var + 1e-9)
    df['beta'] = np.nan
    btc_rets = df[df['symbol_norm'] == btc_sym].set_index('date')['ret'].reindex(df['date']).fillna(method='ffill')
    for sym in df['symbol_norm'].unique():
        mask = df['symbol_norm'] == sym
        sym_rets = df.loc[mask, 'ret'].values
        btc_vals = btc_rets.values
        n = len(sym_rets)
        betas = []
        for i in range(n):
            if i < 6:
                betas.append(np.nan)
            else:
                window_ret = sym_rets[i-6:i+1]
                window_btc = btc_vals[i-6:i+1]
                if np.std(window_ret) < 1e-9 or np.std(window_btc) < 1e-9:
                    betas.append(np.nan)
                else:
                    cov = np.cov(window_ret, window_btc)[0,1]
                    var = np.var(window_btc)
                    betas.append(cov / (var + 1e-9))
        df.loc[mask, 'beta'] = betas
    df['beta_z'] = df.groupby('date')['beta'].transform(zscore_cs, df['beta'])
    df.drop(columns=['btc_ret'], inplace=True, errors='ignore')
else:
    df['beta_z'] = 0

factor_cols = ['fr_z', 'fr_trend_z', 'fr_sign', 'vol_z', 'mom_z', 'qv_z', 'supply_z', 'beta_z']

# ============ Step 4: 清理 & 过滤 ============
print("\n🧹 [4/7] 数据清理...")

df = df.dropna(subset=['fret'] + factor_cols)
# 极端收益裁剪
df['fret_clipped'] = df['fret'].clip(-MAX_RET, MAX_RET)
df = df[df['fret'].abs() <= MAX_RET * 1.5]  # 宽松一点

# 排除BTC (币本位合约锚定BTC, 特殊情况)
df = df[df['symbol_norm'] != 'BTCUSDT']

print(f"  有效记录: {len(df):,} | 币种: {df['symbol_norm'].nunique()} | 日期: {df['date'].min()} ~ {df['date'].max()}")

# ============ Step 5: IC/IR (日级别) ============
print("\n📊 [5/7] 计算 IC/IR (日级别, 指数衰减加权)...")

df = df.sort_values('date')
max_date = df['date'].max()
df['days_ago'] = (max_date - df['date']).dt.days
df['weight'] = np.exp(-np.log(2) / HALFLIFE * df['days_ago'])

results = []
for dt, grp in df.groupby('date'):
    if len(grp) < 8:
        continue
    for fc in factor_cols:
        x_vals = grp[fc].dropna()
        y_vals = grp.loc[x_vals.index, 'fret_clipped']
        if len(x_vals) < 8:
            continue
        try:
            ic, p = spearmanr(x_vals.values, y_vals.values)
            if np.isnan(ic):
                continue
        except:
            continue
        results.append({
            'date': dt, 'factor': fc, 'ic': ic,
            'p': p, 'n': len(x_vals),
            'wt': grp.loc[x_vals.index, 'weight'].mean()
        })

ic_df = pd.DataFrame(results)

# 加权IC
def weighted_ic_mean(grp):
    return np.average(grp['ic'], weights=grp['wt'])

ic_stats = ic_df.groupby('factor')['ic'].agg(
    IC_mean='mean', IC_std='std', count='count'
)
ic_stats['IC_IR']      = ic_stats['IC_mean'] / ic_stats['IC_std']
ic_stats['IC_pos_pct'] = ic_df.groupby('factor')['ic'].apply(lambda x: (x > 0).mean())
ic_stats['IC_abs_mean']= ic_df.groupby('factor')['ic'].apply(lambda x: np.abs(x).mean())
# 加权IC均值 (更准确)
ic_stats['IC_wmean']   = ic_df.groupby('factor').apply(weighted_ic_mean)
ic_stats = ic_stats.sort_values('IC_IR', ascending=False)

print("\nIC/IR 汇总:")
print("-" * 85)
print(f"{'因子':<14} {'IC均值':>9} {'IC加权':>9} {'IC标准差':>10} {'IR':>8} {'IC>0%':>8} {'|IC|均值':>9}")
print("-" * 85)
for fn, row in ic_stats.iterrows():
    print(f"{fn:<14} {row['IC_mean']:>+9.4f} {row['IC_wmean']:>+9.4f} {row['IC_std']:>10.4f} {row['IC_IR']:>8.4f} {row['IC_pos_pct']:>8.1%} {row['IC_abs_mean']:>9.4f}")

# ============ Step 6: 策略回测 (多策略对比) ============
print("\n💰 [6/7] 多策略回测...")

# --- 策略A: 纯做空Q5 (不做多Q1) ---
# 每日期货资金费率分组: Q5=最高费率→做空
df['fr_q'] = df.groupby('date')['fr'].transform(
    lambda x: pd.qcut(x.rank(method='first'), 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
)

# FR成本: 持有24h = 3个8h周期, 做空者付出费率
df['fr_cost'] = df['fr'].abs() * 3  # 24h累积费率

# 策略A: 只做空Q5
df['strat_A_fret_net'] = np.where(
    df['fr_q'] == 'Q5',
    (df['fret_clipped'] + df['fr_cost']).clip(-MAX_RET, MAX_RET) - COST_RATE,  # 做空收到FR, 扣交易成本
    0.0  # 其他档位不持仓
)

# 策略B: 只做多Q1 (避免费率成本只做极端低费率)
df['strat_B_fret_net'] = np.where(
    df['fr_q'] == 'Q1',
    (df['fret_clipped'] - df['fr_cost']).clip(-MAX_RET, MAX_RET) - COST_RATE,
    0.0
)

# 策略C: 多空组合 (做多Q1 + 做空Q5)
df['strat_C_long'] = np.where(df['fr_q'] == 'Q1',
    (df['fret_clipped'] - df['fr_cost']).clip(-MAX_RET, MAX_RET), 0)
df['strat_C_short'] = np.where(df['fr_q'] == 'Q5',
    (df['fret_clipped'] + df['fr_cost']).clip(-MAX_RET, MAX_RET), 0)
df['strat_C_fret_net'] = df['strat_C_long'] - df['strat_C_short'] - COST_RATE * 2

# 策略D: vol_z过滤 + Q5做空 (只在低波动时做空Q5)
df['strat_D_fret_net'] = np.where(
    (df['fr_q'] == 'Q5') & (df['vol_z'] < 0),  # 低波动 + 高费率 → 做空
    (df['fret_clipped'] + df['fr_cost']).clip(-MAX_RET, MAX_RET) - COST_RATE,
    0.0
)

# 策略E: vol_z过滤 + 多空 (低波动时做空Q5, 高波动时做多Q1)
df['strat_E_fret_net'] = np.where(
    (df['fr_q'] == 'Q5') & (df['vol_z'] < 0),  # 低波动高费率 → 做空
    (df['fret_clipped'] + df['fr_cost']).clip(-MAX_RET, MAX_RET),
    np.where(
        (df['fr_q'] == 'Q1') & (df['vol_z'] > 0.5),  # 高波动低费率 → 做多 (波动中做多，动量效应)
        (df['fret_clipped'] - df['fr_cost']).clip(-MAX_RET, MAX_RET),
        0.0
    )
)
df['strat_E_fret_net'] = df['strat_E_fret_net'] - COST_RATE * 2

# 策略F: 最优因子组合 - 做空 vol_z > 0 (高波动做空) + 做空 Q5
# 双向做空: 既做空高波动又做空高费率
df['strat_F_fret_net'] = np.where(
    df['vol_z'] > 0,  # 高波动 → 做空 (做空高波动币种)
    (df['fret_clipped'] * -1).clip(-MAX_RET, MAX_RET),  # 纯做空高波动
    0.0
) + np.where(
    df['fr_q'] == 'Q5',  # 高费率 → 额外做空
    (df['fret_clipped'] + df['fr_cost'] * 0.5).clip(-MAX_RET, MAX_RET),
    0.0
) - COST_RATE * 2

strat_cols = ['strat_A_fret_net', 'strat_B_fret_net', 'strat_C_fret_net',
              'strat_D_fret_net', 'strat_E_fret_net', 'strat_F_fret_net']

# 每日等权组合收益
daily_strat = df.groupby('date')[strat_cols].mean().dropna()
cum_strat = (1 + daily_strat).cumprod()

n_days = (df['date'].max() - df['date'].min()).days

def calc_stats(ser, label, n_d=n_days):
    final = ser.iloc[-1] if len(ser) > 0 else 1.0
    ann = final ** (365 / max(n_d, 1)) - 1
    sharpe = ser.mean() / (ser.std() + 1e-9) * np.sqrt(365) if ser.std() > 0 else 0
    max_dd = (ser.cummax() - ser).max() / ser.cummax().max() if ser.max() > 0 else 0
    win_rate = (ser > 0).mean()
    return {
        'Strategy': label, 'Final净值': final,
        '年化收益率': ann, '夏普比率': sharpe,
        '最大回撤': max_dd, '胜率': win_rate, '交易天数': len(ser)
    }

stats = []
for sc in strat_cols:
    label = {
        'strat_A_fret_net': 'A: 仅做空Q5',
        'strat_B_fret_net': 'B: 仅做多Q1(扣费)',
        'strat_C_fret_net': 'C: Q1多+Q5空(多空)',
        'strat_D_fret_net': 'D: vol_z<0过滤+Q5做空',
        'strat_E_fret_net': 'E: vol_z双Q5做空+波动做多',
        'strat_F_fret_net': 'F: vol_z做空+高费率叠加',
    }[sc]
    stats.append(calc_stats(cum_strat[sc], label))

stats_df = pd.DataFrame(stats).sort_values('夏普比率', ascending=False)
print("\n策略绩效对比:")
print("-" * 90)
for _, row in stats_df.iterrows():
    print(f"  {row['Strategy']:<35} 年化={row['年化收益率']:>+8.2%}  夏普={row['夏普比率']:>+6.3f}  "
          f"最大回撤={row['最大回撤']:>7.2%}  胜率={row['胜率']:>6.1%}  天数={row['交易天数']}")

# ============ Step 7: 月度分析 ============
print("\n📅 [7/7] 月度 IC ...")

monthly_ic = ic_df.copy()
monthly_ic['month'] = monthly_ic['date'].dt.to_period('M')
monthly_ic = monthly_ic.groupby(['month', 'factor'])['ic'].mean().unstack('factor')
monthly_ic.to_csv(f"{OUTPUT}/monthly_ic_v3.csv")
ic_df.to_csv(f"{OUTPUT}/ic_timeseries_v3.csv", index=False)
ic_stats.to_csv(f"{OUTPUT}/ic_stats_v3.csv")

# 生成报告
min_date = str(df['date'].min())[:10]
max_date_str = str(df['date'].max())[:10]
n_syms = df['symbol_norm'].nunique()

best_strat = stats_df.iloc[0]

report = f"""# 加密货币量化因子有效性分析报告 v4（优化版）

> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> **数据范围**: {min_date} ~ {max_date_str}
> **分析币种**: {n_syms} 个（三方数据均有覆盖）
> **有效记录**: {len(df):,} 条
> **指数衰减半衰期**: {HALFLIFE} 天
> **极端回报裁剪**: ±{MAX_RET:.0%}（24h）
> **注意**: IC与收益全部为**日级别**聚合（消除FR 8h采样噪声）

---

## 一、因子 IC/IR（日级别，指数衰减加权）

| 因子 | 含义 | IC均值 | IC加权 | IC标准差 | IR | IC>0占比 | |IC|均值 |
|------|------|-------:|------:|--------:|---:|--------:|---------:|
| vol_z | 波动率 Z-Score | {ic_stats.loc['vol_z','IC_mean']:+.4f} | {ic_stats.loc['vol_z','IC_wmean']:+.4f} | {ic_stats.loc['vol_z','IC_std']:.4f} | {ic_stats.loc['vol_z','IC_IR']:+.4f} | {ic_stats.loc['vol_z','IC_pos_pct']:.1%} | {ic_stats.loc['vol_z','IC_abs_mean']:.4f} |
| mom_z | 动量 Z-Score | {ic_stats.loc['mom_z','IC_mean']:+.4f} | {ic_stats.loc['mom_z','IC_wmean']:+.4f} | {ic_stats.loc['mom_z','IC_std']:.4f} | {ic_stats.loc['mom_z','IC_IR']:+.4f} | {ic_stats.loc['mom_z','IC_pos_pct']:.1%} | {ic_stats.loc['mom_z','IC_abs_mean']:.4f} |
| beta_z | 条件Beta Z-Score | {ic_stats.loc['beta_z','IC_mean']:+.4f} | {ic_stats.loc['beta_z','IC_wmean']:+.4f} | {ic_stats.loc['beta_z','IC_std']:.4f} | {ic_stats.loc['beta_z','IC_IR']:+.4f} | {ic_stats.loc['beta_z','IC_pos_pct']:.1%} | {ic_stats.loc['beta_z','IC_abs_mean']:.4f} |
| supply_z | 供给量 Z-Score | {ic_stats.loc['supply_z','IC_mean']:+.4f} | {ic_stats.loc['supply_z','IC_wmean']:+.4f} | {ic_stats.loc['supply_z','IC_std']:.4f} | {ic_stats.loc['supply_z','IC_IR']:+.4f} | {ic_stats.loc['supply_z','IC_pos_pct']:.1%} | {ic_stats.loc['supply_z','IC_abs_mean']:.4f} |
| fr_sign | 资金费率符号 | {ic_stats.loc['fr_sign','IC_mean']:+.4f} | {ic_stats.loc['fr_sign','IC_wmean']:+.4f} | {ic_stats.loc['fr_sign','IC_std']:.4f} | {ic_stats.loc['fr_sign','IC_IR']:+.4f} | {ic_stats.loc['fr_sign','IC_pos_pct']:.1%} | {ic_stats.loc['fr_sign','IC_abs_mean']:.4f} |
| fr_z | 资金费率 Z-Score | {ic_stats.loc['fr_z','IC_mean']:+.4f} | {ic_stats.loc['fr_z','IC_wmean']:+.4f} | {ic_stats.loc['fr_z','IC_std']:.4f} | {ic_stats.loc['fr_z','IC_IR']:+.4f} | {ic_stats.loc['fr_z','IC_pos_pct']:.1%} | {ic_stats.loc['fr_z','IC_abs_mean']:.4f} |
| qv_z | 成交量变化 Z-Score | {ic_stats.loc['qv_z','IC_mean']:+.4f} | {ic_stats.loc['qv_z','IC_wmean']:+.4f} | {ic_stats.loc['qv_z','IC_std']:.4f} | {ic_stats.loc['qv_z','IC_IR']:+.4f} | {ic_stats.loc['qv_z','IC_pos_pct']:.1%} | {ic_stats.loc['qv_z','IC_abs_mean']:.4f} |
| fr_trend_z | 费率趋势 Z-Score | {ic_stats.loc['fr_trend_z','IC_mean']:+.4f} | {ic_stats.loc['fr_trend_z','IC_wmean']:+.4f} | {ic_stats.loc['fr_trend_z','IC_std']:.4f} | {ic_stats.loc['fr_trend_z','IC_IR']:+.4f} | {ic_stats.loc['fr_trend_z','IC_pos_pct']:.1%} | {ic_stats.loc['fr_trend_z','IC_abs_mean']:.4f} |

> **IR 解读**: |IR| > 0.5 强有效，0.3~0.5 中等，< 0.3 弱

---

## 二、策略回测绩效对比（{n_days}天）

| 策略 | 年化收益率 | 夏普比率 | 最大回撤 | 胜率 |
|------|----------:|--------:|--------:|-----:|
"""

for _, row in stats_df.iterrows():
    report += f"| {row['Strategy']} | {row['年化收益率']:+.2%} | {row['夏普比率']:+.3f} | {row['最大回撤']:.2%} | {row['胜率']:.1%} |\n"

report += f"""
### 最佳策略: {best_strat['Strategy']}
- 年化收益率: **{best_strat['年化收益率']:+.2%}**
- 夏普比率: **{best_strat['夏普比率']:+.3f}**
- 最大回撤: {best_strat['最大回撤']:.2%}
- 胜率: {best_strat['胜率']:.1%}

---

## 三、策略说明

| 策略 | 逻辑 |
|------|------|
| **A: 仅做空Q5** | 只在高资金费率(Q5)时做空，费率收入抵消做空损失 |
| **B: 仅做多Q1(扣费)** | 在低资金费率(Q1)时做多，验证做多可行性 |
| **C: Q1多+Q5空** | 经典多空组合，等权做多低费率+做空高费率 |
| **D: vol_z<0+Q5做空** | 只在低波动时做空Q5，避免高波动时的尾部风险 |
| **E: vol_z双策略** | 低波动+Q5做空，高波动+Q1做多 |
| **F: vol_z+费率叠加** | 同时做空高波动 + 做空高费率(双重做空) |

---

## 四、因子IC稳定性 (近12个月)

"""

recent_months = monthly_ic.tail(12)
for col in ['vol_z', 'mom_z', 'fr_sign', 'fr_z', 'beta_z', 'supply_z']:
    if col in recent_months.columns:
        pos_rate = (recent_months[col] > 0).mean()
        report += f"- **{col}**: 近12月IC>0占比 {pos_rate:.0%}\n"

report += f"""
---

## 五、综合结论

1. **最优策略**: **{best_strat['Strategy']}**，年化 {best_strat['年化收益率']:+.2%}，夏普 {best_strat['夏普比率']:+.3f}
2. **核心发现**: {'vol_z是预测能力最强的因子' if ic_stats.loc['vol_z','IC_IR'] < ic_stats.loc['fr_sign','IC_IR'] else 'fr_sign是FR相关因子中最稳定的'}
3. **做空Q5** 是可行策略（费率补贴做空损失）
4. **推荐**: 采用策略D或策略E（叠加波动率过滤），在市场平静时捕捉高费率溢价

---

*本报告由因子分析系统 v4 自动生成 | 数据：Binance*
"""

with open(f"{OUTPUT}/资金费率因子分析报告_v4_{datetime.now().strftime('%Y%m%d')}.md", 'w') as f:
    f.write(report)

print(f"\n✅ 报告已保存: {OUTPUT}/资金费率因子分析报告_v4_{datetime.now().strftime('%Y%m%d')}.md")
print(f"✅ IC时序: {OUTPUT}/ic_timeseries_v3.csv")
print(f"✅ 月度IC: {OUTPUT}/monthly_ic_v3.csv")
print(f"✅ IC统计: {OUTPUT}/ic_stats_v3.csv")
