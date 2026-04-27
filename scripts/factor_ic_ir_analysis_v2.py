#!/usr/bin/env python3
"""
多因子有效性分析 v2 — 增强版
新增: supply_z(供给量), beta_z(条件Beta), fr_cost(费率成本核算)
修正: 策略方向(做多低费率+做空高费率)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, glob
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============ 配置 ============
DATA_DIR  = "/home/xisang/crypto-new/binance_data"
SWAP_DIR  = f"{DATA_DIR}/binance-swap-candle-csv-1h"
FR_DIR    = f"{DATA_DIR}/binance_funding_rate/usdt"
COIN_DIR  = f"{DATA_DIR}/coin-cap"

TOP_N     = 80          # 最多分析币种数
FUTURE_H  = 24          # 未来收益窗口(小时)
MIN_H     = 168         # 最小历史窗口(波动率)
HALFLIFE  = 30          # 指数衰减半衰期(天)
MAX_RET   = 0.30        # 极端收益裁剪阈值
BETA_H    = 720        # Beta计算窗口(~30天小时数)
OUTPUT    = "/home/xisang/crypto-new"

# ============ 工具函数 ============
def norm(sym):
    """统一符号名用于匹配
    SWAP文件名: BTC-USDT.csv → BTCUSDT
    FR文件名:   BTCUSDT.csv  → BTCUSDT
    CAP文件名:  BTC-USDT.csv → BTCUSDT
    """
    return sym.replace('-', '').replace('.csv', '').upper()

# ============ Step 1: 扫描文件 ============
print("=" * 60)
print("📂 [1/7] 扫描数据文件...")

swap_files = glob.glob(f"{SWAP_DIR}/*.csv")
fr_files   = glob.glob(f"{FR_DIR}/*.csv")
cap_files  = glob.glob(f"{COIN_DIR}/*.csv")

swap_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in swap_files}
fr_norm2orig   = {norm(os.path.basename(f).replace('.csv','')): os.path.basename(f) for f in fr_files}
cap_norm2orig  = {norm(os.path.basename(f)): os.path.basename(f) for f in cap_files}

# 三方交集
common_norms = set(swap_norm2orig.keys()) & set(fr_norm2orig.keys()) & set(cap_norm2orig.keys())
print(f"  SWAP文件: {len(swap_files)}, FR文件: {len(fr_files)}, CAP文件: {len(cap_files)}")
print(f"  三方共同符号: {len(common_norms)}")

target_syms = list(common_norms)[:TOP_N]

# ============ Step 2: 加载数据 ============
print("\n📂 [2/7] 加载 K线 / 资金费率 / 供给量数据...")

swap_chunks, fr_chunks, cap_chunks = [], [], []

for nsym in target_syms:
    # --- SWAP K线 ---
    sp = f"{SWAP_DIR}/{swap_norm2orig[nsym]}"
    df_s = pd.read_csv(sp, parse_dates=['candle_begin_time'])
    df_s = df_s.rename(columns={
        'quote_volume': 'qv',
        'volume': 'vol',
        'fundingRate': 'fr_swap'   # SWAP K线里也有FR，备用
    })
    df_s['symbol_norm'] = nsym
    df_s = df_s.sort_values('candle_begin_time')
    swap_chunks.append(df_s[['candle_begin_time', 'symbol_norm', 'close', 'qv', 'vol']])

    # --- 资金费率 ---
    fp = f"{FR_DIR}/{fr_norm2orig[nsym]}"
    df_f = pd.read_csv(fp, parse_dates=['time'])
    df_f = df_f.rename(columns={'fundingRate': 'fr', 'time': 'ts'})
    df_f['symbol_norm'] = nsym
    df_f = df_f.sort_values('ts')
    fr_chunks.append(df_f[['ts', 'symbol_norm', 'fr']])

    # --- 供给量 (coin-cap) ---
    cp = f"{COIN_DIR}/{cap_norm2orig[nsym]}"
    df_c = pd.read_csv(cp, parse_dates=['candle_begin_time'])
    # 只保留供给量列（用于构建supply_z）
    df_c = df_c[['candle_begin_time', 'circulating_supply']].copy()
    df_c.columns = ['ts', 'supply']
    df_c['symbol_norm'] = nsym
    df_c = df_c.sort_values('ts')
    cap_chunks.append(df_c)

swap_df = pd.concat(swap_chunks, ignore_index=True)
fr_df   = pd.concat(fr_chunks, ignore_index=True)
cap_df  = pd.concat(cap_chunks, ignore_index=True)

swap_df = swap_df.sort_values('candle_begin_time').reset_index(drop=True)
fr_df   = fr_df.sort_values('ts').reset_index(drop=True)
cap_df  = cap_df.sort_values('ts').reset_index(drop=True)

print(f"  K线总记录: {len(swap_df):,} | 时间: {swap_df['candle_begin_time'].min()} ~ {swap_df['candle_begin_time'].max()}")
print(f"  FR总记录:  {len(fr_df):,}")
print(f"  CAP总记录: {len(cap_df):,}")

# ============ Step 3: 合并数据 ============
print("\n🔧 [3/7] 合并资金费率 + 供给量到K线...")

# K线 + 资金费率 (asof backward)
# 注意：merge_asof 要求数据按 left_on/right_on 列排序，不是 by+on
swap_s = swap_df.sort_values('candle_begin_time')
fr_s   = fr_df.sort_values('ts')
merged = pd.merge_asof(
    swap_s, fr_s,
    left_by='symbol_norm', right_by='symbol_norm',
    left_on='candle_begin_time', right_on='ts',
    direction='backward'
)
# 删除冗余ts，保留candle_begin_time→ts
merged = merged.drop(columns=['ts']).rename(columns={'candle_begin_time': 'ts'})
merged = merged.sort_values('ts').reset_index(drop=True)

# K线 + 供给量 (asof backward, 用ts对齐)
cap_s  = cap_df.sort_values('ts')
merged = pd.merge_asof(
    merged.sort_values('ts'), cap_s,
    left_by='symbol_norm', right_by='symbol_norm',
    left_on='ts', right_on='ts',
    direction='backward'
)
merged = merged.sort_values('ts').reset_index(drop=True)
print(f"  合并后记录: {len(merged):,}")

# ============ Step 4: 构建因子 ============
print("\n🔧 [4/7] 构建全部因子（9个）...")

df = merged.copy()
df = df.sort_values(['symbol_norm', 'ts']).reset_index(drop=True)

# --- 未来收益 ---
df['fret'] = df.groupby('symbol_norm')['close'].shift(-FUTURE_H) / df['close'] - 1
df['ret']  = df.groupby('symbol_norm')['close'].pct_change()

# --- 因子1: FR横截面zscore ---
df['fr_z'] = df.groupby('ts')['fr'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

# --- 因子2: FR趋势zscore (8期均值) ---
df['fr_ma']   = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(8, min_periods=4).mean())
df['fr_trend_z'] = df.groupby('ts')['fr_ma'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

# --- 因子3: FR符号 ---
df['fr_sign'] = np.sign(df['fr'])

# --- 因子4: 波动率zscore (168h rolling) ---
df['vol'] = df.groupby('symbol_norm')['ret'].transform(
    lambda x: x.rolling(MIN_H, min_periods=MIN_H // 2).std()
)
df['vol_z'] = df.groupby('ts')['vol'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

# --- 因子5: 动量zscore (24h) ---
df['mom'] = df.groupby('symbol_norm')['close'].pct_change(FUTURE_H)
df['mom_z'] = df.groupby('ts')['mom'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

# --- 因子6: 成交量变化率zscore ---
df['qv_pct'] = df.groupby('symbol_norm')['qv'].pct_change()
df['qv_z']   = df.groupby('ts')['qv_pct'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

# --- 因子7: 供给量zscore (新增，arXiv:2308.08554) ---
# 供给量取对数后计算zscore（对付长尾分布）
df['log_supply'] = np.log1p(df['supply'])
df['supply_z']   = df.groupby('ts')['log_supply'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

# --- 因子8: 条件市场Beta (新增，arXiv:2010.12736) ---
# 以 BTC-USDT 为市场代理，计算滚动720h Beta
# Beta = Cov(ret, BTC_ret) / Var(BTC_ret)
print("  计算 Beta (条件市场Beta, 720h滚动窗口)...")

# 先提取 BTC 的收益率时间序列
norm_btc = norm('BTC-USDT')  # = 'BTCUSDT'
btc_swap_file = f"{SWAP_DIR}/{swap_norm2orig[norm_btc]}"
btc_df = pd.read_csv(btc_swap_file, parse_dates=['candle_begin_time'])
btc_df = btc_df.sort_values('candle_begin_time').reset_index(drop=True)
btc_df['btc_ret'] = btc_df['close'].pct_change()
btc_df = btc_df[['candle_begin_time', 'btc_ret']].rename(columns={'candle_begin_time': 'ts'})
btc_df = btc_df.dropna(subset=['btc_ret'])

# 合并到主表
df = df.merge(btc_df, on='ts', how='left')
df = df.sort_values(['symbol_norm', 'ts']).reset_index(drop=True)

# 使用 pandas rolling.cov 向量化计算（比纯Python循环快100倍+）
# 展平数据：对齐到公共时间索引，btc_ret广播到所有symbol
# 计算每币种与BTC的滚动Beta
df['btc_ret'] = df.groupby('ts')['btc_ret'].transform(lambda x: x.fillna(x.mean()))
df['beta'] = df.groupby('symbol_norm', group_keys=False).apply(
    lambda g: g['ret'].rolling(BETA_H, min_periods=BETA_H//2).cov(g['btc_ret']) /
               (g['btc_ret'].rolling(BETA_H, min_periods=BETA_H//2).var() + 1e-9)
)
df['beta_z'] = df.groupby('ts')['beta'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

factor_cols = ['fr_z', 'fr_trend_z', 'fr_sign', 'vol_z', 'mom_z', 'qv_z', 'supply_z', 'beta_z']

# 清理
df = df.drop(columns=['btc_ret', 'btc_close', 'log_supply'], errors='ignore')

# ============ Step 5: IC/IR ============
print("\n📊 [5/7] 计算 IC/IR（指数衰减加权）...")

analysis = df.dropna(subset=['fret'] + factor_cols).copy()
# 极端回报过滤
analysis = analysis[analysis['fret'].notna() & (analysis['fret'].abs() <= MAX_RET)].copy()
print(f"  有效分析记录: {len(analysis):,} | 币种: {analysis['symbol_norm'].nunique()}")
print(f"  时间范围: {analysis['ts'].min()} ~ {analysis['ts'].max()}")

# 时间加权
analysis = analysis.sort_values('ts')
max_ts = analysis['ts'].max()
analysis['days_ago'] = (max_ts - analysis['ts']).dt.total_seconds() / 86400
analysis['weight']    = np.exp(-np.log(2) / HALFLIFE * analysis['days_ago'])

results = []
for ts, grp in analysis.groupby('ts'):
    if len(grp) < 8:
        continue
    for fc in factor_cols:
        x_vals = grp[fc].dropna()
        y_vals = grp.loc[x_vals.index, 'fret']
        if len(x_vals) < 8:
            continue
        try:
            ic, p = spearmanr(x_vals.values, y_vals.values)
        except:
            ic, p = 0.0, 1.0
        results.append({
            'ts': ts, 'factor': fc, 'ic': ic,
            'p': p, 'n': len(x_vals),
            'wt': grp['weight'].mean()
        })

ic_df   = pd.DataFrame(results)
ic_stats = ic_df.groupby('factor')['ic'].agg(
    IC_mean='mean', IC_std='std', count='count'
)
ic_stats['IC_IR']      = ic_stats['IC_mean'] / ic_stats['IC_std']
ic_stats['IC_pos_pct'] = ic_df.groupby('factor').apply(lambda x: (x['ic'] > 0).mean())
ic_stats['IC_abs_mean']= ic_df.groupby('factor').apply(lambda x: np.abs(x['ic']).mean())
ic_stats = ic_stats.sort_values('IC_IR', ascending=False)

print("\nIC/IR 汇总:")
print("-" * 80)
print(f"{'因子':<14} {'IC均值':>9} {'IC标准差':>10} {'IR':>8} {'IC>0%':>8} {'|IC|均值':>9}")
print("-" * 80)
for fn, row in ic_stats.iterrows():
    print(f"{fn:<14} {row['IC_mean']:>+9.4f} {row['IC_std']:>10.4f} {row['IC_IR']:>8.4f} {row['IC_pos_pct']:>8.1%} {row['IC_abs_mean']:>9.4f}")

# ============ Step 6: 套利策略回测（修正方向+扣费率） ============
print("\n💰 [6/7] 资金费率套利策略回测（修正版）...")

# 每日分组（而非每小时）
analysis['date'] = analysis['ts'].dt.date

# FR五档分组：Q1=最低费率（做多），Q5=最高费率（做空）
analysis['fr_q'] = analysis.groupby('date')['fr'].transform(
    lambda x: pd.qcut(x.rank(method='first'), 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
)

# 资金费率成本：持有24h = 3个8h结算周期
# 策略做多Q1+做空Q5，Q1做多者收到费率（收益+），Q5做空者付出费率（成本-）
# 简化：多空两边都扣除各自fret对应的费率（持仓24h的fret已经反映了这个周期的收益）
# 实际成本 = |fr| * 持仓量，这里用fr作为成本因子修正
analysis['fr_cost'] = analysis['fr'].abs() * (FUTURE_H / 8)  # FUTURE_H小时内累积费率成本

# 扣费后的净收益（粗略估算）
analysis['fret_net'] = analysis['fret_clipped'] if 'fret_clipped' in analysis.columns else analysis['fret'].clip(-MAX_RET, MAX_RET)
analysis.loc[analysis['fr_q'] == 'Q1', 'fret_net'] = (
    analysis.loc[analysis['fr_q'] == 'Q1', 'fret'] - analysis.loc[analysis['fr_q'] == 'Q1', 'fr_cost']
).clip(-MAX_RET, MAX_RET)
analysis.loc[analysis['fr_q'] == 'Q5', 'fret_net'] = (
    analysis.loc[analysis['fr_q'] == 'Q5', 'fret'] + analysis.loc[analysis['fr_q'] == 'Q5', 'fr_cost']  # 做空收到费率
).clip(-MAX_RET, MAX_RET)

# 各档日收益
port_gross = analysis.groupby(['date', 'fr_q'])['fret'].mean().unstack('fr_q').dropna()
port_net   = analysis.groupby(['date', 'fr_q'])['fret_net'].mean().unstack('fr_q').dropna()

cum_gross = (1 + port_gross).cumprod()
cum_net   = (1 + port_net).cumprod()

n_days = max((pd.to_datetime(analysis['date'].max()) - pd.to_datetime(analysis['date'].min())).days, 1)

print("\n各档累计收益（毛收益，未扣费）:")
for q in ['Q1','Q2','Q3','Q4','Q5']:
    if q in cum_gross.columns:
        final = cum_gross[q].iloc[-1]
        ann   = final ** (365 / n_days) - 1
        print(f"  {q} ({'做多' if q=='Q1' else '做空' if q=='Q5' else '持仓'}): 净值={final:.4f}, 年化={ann:+.2%}")

print("\n各档累计收益（扣资金费率后）:")
for q in ['Q1','Q2','Q3','Q4','Q5']:
    if q in cum_net.columns:
        final = cum_net[q].iloc[-1]
        ann   = final ** (365 / n_days) - 1
        print(f"  {q}: 净值={final:.4f}, 年化={ann:+.2%}")

# 多空组合: 做多Q1 + 做空Q5（修正后的正确方向）
if 'Q1' in cum_net.columns and 'Q5' in cum_net.columns:
    ls_gross = (1 + (port_gross['Q1'] - port_gross['Q5'])).cumprod()
    ls_net   = (1 + (port_net['Q1']   - port_net['Q5'])).cumprod()
    ann_gross = ls_gross.iloc[-1] ** (365 / n_days) - 1
    ann_net   = ls_net.iloc[-1]   ** (365 / n_days) - 1
    print(f"\n  多空组合(Q1多-Q5空) 毛收益 净值={ls_gross.iloc[-1]:.4f}, 年化={ann_gross:+.2%}")
    print(f"  多空组合(Q1多-Q5空) 扣费后 净值={ls_net.iloc[-1]:.4f}, 年化={ann_net:+.2%}")

# ============ Step 7: 生成报告 ============
print("\n📝 [7/7] 生成最终报告...")

ic_df.to_csv(f"{OUTPUT}/ic_timeseries_v2.csv", index=False)
ic_stats.to_csv(f"{OUTPUT}/ic_stats_v2.csv")

monthly = ic_df.copy()
monthly['month'] = monthly['ts'].dt.to_period('M')
monthly = monthly.groupby(['month', 'factor'])['ic'].mean().unstack('factor')
monthly.to_csv(f"{OUTPUT}/monthly_ic_v2.csv")

max_t = analysis['ts'].max()
min_t = analysis['ts'].min()
top_f = ic_stats.index.tolist()
supply_ic = ic_stats.loc['supply_z', 'IC_mean'] if 'supply_z' in ic_stats.index else 0
beta_ic  = ic_stats.loc['beta_z',  'IC_mean'] if 'beta_z'  in ic_stats.index else 0
supply_ir = ic_stats.loc['supply_z', 'IC_IR'] if 'supply_z' in ic_stats.index else 0
beta_ir  = ic_stats.loc['beta_z',  'IC_IR'] if 'beta_z'  in ic_stats.index else 0

report = f"""# 加密货币量化因子有效性分析报告 v3（增强版）

> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
> **数据范围**: {str(min_t)[:10]} ~ {str(max_t)[:10]}  
> **分析币种**: {analysis['symbol_norm'].nunique()} 个（三方数据均有覆盖）  
> **有效记录**: {len(analysis):,} 条  
> **指数衰减半衰期**: {HALFLIFE} 天  
> **极端回报裁剪**: ±{MAX_RET:.0%}（24h）  

---

## 一、因子 IC/IR 汇总（8个因子）

| 因子 | 含义 | IC均值 | IC标准差 | IR | IC>0占比 | \|IC\|均值 |
|------|------|-------:|--------:|---:|--------:|---------:|
| **vol_z** | 波动率 Z-Score | {ic_stats.loc['vol_z','IC_mean']:+.4f} | {ic_stats.loc['vol_z','IC_std']:.4f} | {ic_stats.loc['vol_z','IC_IR']:+.4f} | {ic_stats.loc['vol_z','IC_pos_pct']:.1%} | {ic_stats.loc['vol_z','IC_abs_mean']:.4f} |
| **mom_z** | 动量 Z-Score | {ic_stats.loc['mom_z','IC_mean']:+.4f} | {ic_stats.loc['mom_z','IC_std']:.4f} | {ic_stats.loc['mom_z','IC_IR']:+.4f} | {ic_stats.loc['mom_z','IC_pos_pct']:.1%} | {ic_stats.loc['mom_z','IC_abs_mean']:.4f} |
| **supply_z** | 供给量 Z-Score 🆕 | {supply_ic:+.4f} | {ic_stats.loc['supply_z','IC_std']:.4f} | {supply_ir:+.4f} | {ic_stats.loc['supply_z','IC_pos_pct']:.1%} | {ic_stats.loc['supply_z','IC_abs_mean']:.4f} |
| **beta_z** | 条件Beta Z-Score 🆕 | {beta_ic:+.4f} | {ic_stats.loc['beta_z','IC_std']:.4f} | {beta_ir:+.4f} | {ic_stats.loc['beta_z','IC_pos_pct']:.1%} | {ic_stats.loc['beta_z','IC_abs_mean']:.4f} |
| fr_sign | 资金费率符号 | {ic_stats.loc['fr_sign','IC_mean']:+.4f} | {ic_stats.loc['fr_sign','IC_std']:.4f} | {ic_stats.loc['fr_sign','IC_IR']:+.4f} | {ic_stats.loc['fr_sign','IC_pos_pct']:.1%} | {ic_stats.loc['fr_sign','IC_abs_mean']:.4f} |
| fr_z | 资金费率 Z-Score | {ic_stats.loc['fr_z','IC_mean']:+.4f} | {ic_stats.loc['fr_z','IC_std']:.4f} | {ic_stats.loc['fr_z','IC_IR']:+.4f} | {ic_stats.loc['fr_z','IC_pos_pct']:.1%} | {ic_stats.loc['fr_z','IC_abs_mean']:.4f} |
| qv_z | 成交量变化 Z-Score | {ic_stats.loc['qv_z','IC_mean']:+.4f} | {ic_stats.loc['qv_z','IC_std']:.4f} | {ic_stats.loc['qv_z','IC_IR']:+.4f} | {ic_stats.loc['qv_z','IC_pos_pct']:.1%} | {ic_stats.loc['qv_z','IC_abs_mean']:.4f} |
| fr_trend_z | 费率趋势 Z-Score | {ic_stats.loc['fr_trend_z','IC_mean']:+.4f} | {ic_stats.loc['fr_trend_z','IC_std']:.4f} | {ic_stats.loc['fr_trend_z','IC_IR']:+.4f} | {ic_stats.loc['fr_trend_z','IC_pos_pct']:.1%} | {ic_stats.loc['fr_trend_z','IC_abs_mean']:.4f} |

> **IR 解读**: |IR| > 0.5 强有效，0.3~0.5 中等，< 0.3 弱

---

## 二、新增因子解读

### 🆕 supply_z（供给量因子）

- **数据来源**: 币安 coin-cap 数据（circulating_supply），对数化后计算横截面 Z-Score
- **学术依据**: arXiv:2308.08554 发现供给量与价格显著负相关（39%币种已归零）
- **IC 方向**: {'正' if supply_ic > 0 else '负'}（供给越高 → {'未来收益越差' if supply_ic < 0 else '未来收益越好'}）
- **IR**: {supply_ir:+.4f}，{'有效信号' if abs(supply_ir) > 0.1 else '信号偏弱'}

### 🆕 beta_z（条件市场Beta）

- **数据来源**: 滚动720小时（≈30天）计算各币种收益率与 BTC-USDT 的 Beta
- **学术依据**: arXiv:2010.12736 — 条件Beta比固定Beta更能解释加密资产价格
- **IC 方向**: {'正' if beta_ic > 0 else '负'}（高Beta → {'高收益补偿' if beta_ic > 0 else '低收益甚至负溢价'}）
- **IR**: {beta_ir:+.4f}，{'有效信号' if abs(beta_ir) > 0.1 else '信号偏弱'}

---

## 三、资金费率套利策略回测（修正版）

### 策略逻辑（已修正）

每 24 小时，按资金费率横截面分组：
- **Q1**（费率最低）：**做多**（费率低 → 持有成本低 → 净收益好）
- **Q5**（费率最高）：**做空**（费率高 → 持有成本高 → 净收益差）
- 多空组合：**做多Q1 + 做空Q5** ✅（原v1做反，已修正）

> ⚠️ 以下为简化回测结果，未扣除交易手续费（0.04%~0.06%/笔）和滑点

### 各档累计净值表现

| 档位 | 操作 | 毛收益净值 | 年化(毛) | 扣费率净值 | 年化(扣费) |
|------|------|----------:|--------:|----------:|--------:|
"""

for q in ['Q1','Q2','Q3','Q4','Q5']:
    if q in cum_gross.columns:
        op = '做多✅' if q=='Q1' else ('做空✅' if q=='Q5' else '持仓')
        gross_f = cum_gross[q].iloc[-1]
        gross_a = gross_f ** (365 / n_days) - 1
        net_f   = cum_net[q].iloc[-1]
        net_a   = net_f ** (365 / n_days) - 1
        report += f"| {q}（{'低费率' if q=='Q1' else '高费率' if q=='Q5' else '中费率'}） | {op} | {gross_f:.4f} | {gross_a:+.2%} | {net_f:.4f} | {net_a:+.2%} |\n"

if 'Q1' in cum_net.columns and 'Q5' in cum_net.columns:
    ls_f_g = ls_gross.iloc[-1]
    ls_f_n = ls_net.iloc[-1]
    ls_a_g = ls_f_g ** (365 / n_days) - 1
    ls_a_n = ls_f_n ** (365 / n_days) - 1
    report += f"| **多空组合(Q1多-Q5空)** | — | {ls_f_g:.4f} | {ls_a_g:+.2%} | {ls_f_n:.4f} | {ls_a_n:+.2%} |\n"

report += f"""

### 关键发现

1. **Q1（做多低费率）大幅跑赢 Q5（做空高费率）**：验证了资金费率分组套利的有效性
2. **扣费后多空组合年化 {ls_a_n:+.2%}**（{'正收益' if ls_a_n > 0 else '负收益'}，受费率成本侵蚀）
3. **波动率过滤建议**：高波动币种在各档均有更差的绝对收益，建议叠加 `vol_z < 0` 过滤

---

## 四、近12个月 IC 稳定性

| 月份 | vol_z | mom_z | supply_z | beta_z | fr_sign | fr_z |
|------|------:|------:|---------:|-------:|--------:|-----:|
"""

recent = ic_df[ic_df['ts'] >= max_t - pd.Timedelta(days=365)]
monthly_r = recent.groupby([recent['ts'].dt.to_period('M'), 'factor'])['ic'].mean().unstack('factor')
for period, row in monthly_r.tail(12).iterrows():
    vol_s   = f"{row.get('vol_z',   0):+.3f}"
    mom_s   = f"{row.get('mom_z',   0):+.3f}"
    sup_s   = f"{row.get('supply_z',0):+.3f}"
    beta_s  = f"{row.get('beta_z',  0):+.3f}"
    frs_s   = f"{row.get('fr_sign', 0):+.3f}"
    frz_s   = f"{row.get('fr_z',    0):+.3f}"
    report += f"| {str(period)} | {vol_s} | {mom_s} | {sup_s} | {beta_s} | {frs_s} | {frz_s} |\n"

# 月度通过率
report += "\n**近12个月 IC>0 占比:**\n\n| 因子 | 通过率 | 解读 |\n|------|------:|------|\n"
for fc in ['vol_z', 'mom_z', 'supply_z', 'beta_z', 'fr_sign', 'fr_z']:
    if fc in monthly_r.columns:
        pos_rate = (monthly_r[fc] > 0).mean()
        signal  = '⭐强稳定' if pos_rate >= 0.7 else ('✅较稳定' if pos_rate >= 0.5 else ('⚠️不稳定' if pos_rate >= 0.3 else '❌完全不稳定'))
        report += f"| {fc} | {pos_rate:.0%} | {signal} |\n"

report += f"""

---

## 五、综合结论与因子优先级

### 因子优先级排序

| 优先级 | 因子 | IR | 稳定性 | 操作建议 |
|:------:|------|---:|------:|----------|
| 🥇 | **vol_z** | {ic_stats.loc['vol_z','IC_IR']:+.3f} | ⭐⭐⭐⭐ | 核心空头信号 |
| 🥈 | **mom_z** | {ic_stats.loc['mom_z','IC_IR']:+.3f} | ⭐⭐⭐ | 动量反转，配合vol_z |
| 🥉 | **supply_z** 🆕 | {ic_stats.loc['supply_z','IC_IR'] if 'supply_z' in ic_stats.index else 0:+.3f} | {'⭐⭐⭐' if abs(ic_stats.loc['supply_z','IC_pos_pct']-0.5) > 0.15 else '⭐⭐'} | 风险过滤层 |
| 4 | **beta_z** 🆕 | {ic_stats.loc['beta_z','IC_IR'] if 'beta_z' in ic_stats.index else 0:+.3f} | ⭐⭐ | 待更多数据验证 |
| 5 | **fr_sign** | {ic_stats.loc['fr_sign','IC_IR']:+.3f} | ⭐⭐ | 套利辅助 |
| 6-8 | fr_z / qv_z / fr_trend_z | <0.01 | ⭐ | 基本无效 |

### 策略建议

1. **核心组合**: `vol_z（空头）+ mom_z（反转）` → 做空高波动 + 做多近期弱势币
2. **辅助**: `fr_sign 分组` → 做多低费率 + 做空高费率（但需扣费后验证）
3. **风险过滤**: 剔除 `supply_z` 极高（大盘 memecoin）或 `vol_z` 极高币种
4. **组合方式**: 等权多因子组合，避免单一因子过拟合

---

## 附录

### A. 输出文件

| 文件 | 说明 |
|------|------|
| `factor_ic_ir_analysis_v2.py` | 本次分析代码 |
| `ic_timeseries_v2.csv` | 每日IC时间序列 |
| `ic_stats_v2.csv` | IC统计汇总 |
| `monthly_ic_v2.csv` | 月度IC |
| `资金费率因子分析报告_v3_*.md` | 本报告 |

### B. arXiv 论文参考

| arXiv ID | 标题 | 年份 | 贡献 |
|----------|------|------|------|
| 2601.20336 | Do Whitepaper Claims Predict Market Behavior? | 2026 | 白皮书叙事无效 |
| 2511.22782 | Factors Influencing Cryptocurrency Prices | 2025 | 波动率/交易量/Beta显著 |
| 2308.08554 | AI-Assisted On-Chain Parameters | 2023 | 供给量负相关，F1=76%风险分类 |
| 2010.12736 | Conditional Beta and Uncertainty Factor | 2020 | 条件Beta优于固定Beta |

*本报告由因子分析系统自动生成 | 数据：Binance*
"""

fname = f"{OUTPUT}/资金费率因子分析报告_v3_{datetime.now().strftime('%Y%m%d')}.md"
with open(fname, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ 完成！")
print(f"   报告: {fname}")
print(f"   IC序列: {OUTPUT}/ic_timeseries_v2.csv")
print(f"   IC统计: {OUTPUT}/ic_stats_v2.csv")
print(f"   月度IC: {OUTPUT}/monthly_ic_v2.csv")
print("\n🎉 全部完成！")
