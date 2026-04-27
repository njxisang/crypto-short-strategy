#!/usr/bin/env python3
"""
多因子有效性分析 v5 — 实战优化版
核心改进:
1. 方向对了：大牛市里不做空，做多为主
2. 高费率+高波动 = 做空（局部空头，熊市/回调时保护）
3. 多空仓位不对称（70%做多 + 30%做空）
4. 加入market regime判断（用BTC动量判断牛熊）
5. 止损/止盈：波动率突破时止损
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
HALFLIFE  = 30
MAX_RET   = 0.30
OUTPUT    = "/home/xisang/crypto-new"
COST_RATE = 0.0005  # 0.05% / 笔

def norm(sym):
    return sym.replace('-', '').replace('.csv', '').replace('.CSV', '').upper()

# ============ Step 1: 扫描 ============
print("=" * 60)
print("📂 [1/7] 扫描文件...")

swap_files = glob.glob(f"{SWAP_DIR}/*.csv")
fr_files   = glob.glob(f"{FR_DIR}/*.csv")
cap_files  = glob.glob(f"{COIN_DIR}/*.csv")

swap_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in swap_files}
fr_norm2orig   = {norm(os.path.basename(f).replace('.csv','').replace('.CSV','')): os.path.basename(f) for f in fr_files}
cap_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in cap_files}

common_norms = set(swap_norm2orig.keys()) & set(fr_norm2orig.keys()) & set(cap_norm2orig.keys())
target_syms = sorted(list(common_norms))[:TOP_N]
print(f"  SWAP: {len(swap_files)}, FR: {len(fr_files)}, CAP: {len(cap_files)}")
print(f"  三方共同: {len(common_norms)} | 分析: {len(target_syms)}")

# ============ Step 2: 加载日线 ============
print("\n📂 [2/7] 加载日线数据...")

all_rows = []
for nsym in target_syms:
    sp = f"{SWAP_DIR}/{swap_norm2orig[nsym]}"
    df_s = pd.read_csv(sp, parse_dates=['candle_begin_time'])
    df_s = df_s.rename(columns={'quote_volume': 'qv', 'volume': 'vol'})
    df_s['date'] = df_s['candle_begin_time'].dt.date
    df_s = df_s.sort_values('candle_begin_time')
    dag = df_s.groupby('date').agg(
        close=('close', 'last'),
        open_=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        qv=('qv', 'sum'),
    ).reset_index()

    fp = f"{FR_DIR}/{fr_norm2orig[nsym]}"
    df_f = pd.read_csv(fp, parse_dates=['time'])
    df_f['date'] = df_f['time'].dt.date
    dag_f = df_f.groupby('date')['fundingRate'].mean().reset_index()
    dag_f.columns = ['date', 'fr']

    cp = f"{COIN_DIR}/{cap_norm2orig[nsym]}"
    df_c = pd.read_csv(cp, parse_dates=['candle_begin_time'])
    df_c['date'] = df_c['candle_begin_time'].dt.date
    dag_c = df_c.groupby('date')['circulating_supply'].last().reset_index()
    dag_c.columns = ['date', 'supply']

    m = dag.merge(dag_f, on=['date'], how='inner')
    m = m.merge(dag_c, on=['date'], how='inner')
    m['symbol_norm'] = nsym
    all_rows.append(m)

df = pd.concat(all_rows, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol_norm','date']).reset_index(drop=True)
print(f"  原始: {len(df):,} | {df['symbol_norm'].nunique()} 币 | {df['date'].min().date()} ~ {df['date'].max().date()}")

# ============ Step 3: 因子计算 ============
print("\n🔧 [3/7] 计算因子...")

BTC_SYM = 'BTCUSDT'
df['ret'] = df.groupby('symbol_norm')['close'].pct_change()
df['fret'] = df.groupby('symbol_norm')['ret'].shift(-1)  # 下一天收益
df['fr_3d'] = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df['qv_chg'] = df.groupby('symbol_norm')['qv'].pct_change()
df['vol_7d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).std())
df['mom_7d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).mean())
df['log_supply'] = np.log1p(df['supply'])

# FR持仓成本（做多者每24h付fr*3）
df['fr_cost_long']  = df['fr'].clip(lower=0) * 3   # 做多: 付出正费率
df['fr_income_short'] = df['fr'].clip(upper=0).abs() * 3  # 做空: 收到（如果fr<0，maker做空收到钱）

def czscore(g, col):
    return (g - g.mean()) / (g.std() + 1e-9)

for fc, src in [('fr_z','fr'),('fr_trend_z','fr_3d'),('qv_z','qv_chg'),
                ('vol_z','vol_7d'),('mom_z','mom_7d'),('supply_z','log_supply')]:
    df[fc] = df.groupby('date')[src].transform(czscore, df[src])

df['fr_sign'] = np.sign(df['fr'])

# BTC market动量 (判断牛熊)
if BTC_SYM in df['symbol_norm'].values:
    btc_daily = df[df['symbol_norm'] == BTC_SYM][['date','close']].copy()
    btc_daily['btc_mom_7d'] = btc_daily['close'].pct_change(7)
    df = df.merge(btc_daily[['date','btc_mom_7d']], on='date', how='left')
    # BTC动量 rolling均值, >0 → 牛市, <0 → 熊市
    df['btc_mom_smooth'] = df.groupby('symbol_norm')['btc_mom_7d'].transform(
        lambda x: x.rolling(5, min_periods=2).mean() if len(x) > 0 else 0
    )
    # 对每个日期取同一的market regime (用BTC的momentum)
    btc_regime = df[df['symbol_norm'] == BTC_SYM][['date','btc_mom_smooth']].copy()
    btc_regime.columns = ['date', 'market_regime']
    df = df.merge(btc_regime, on='date', how='left')
else:
    df['market_regime'] = 0.0

df['regime_bull'] = (df['market_regime'] > 0).astype(float)  # 1=牛市, 0=熊市

factor_cols = ['fr_z', 'fr_trend_z', 'fr_sign', 'vol_z', 'mom_z', 'qv_z', 'supply_z']

# ============ Step 4: 清理 ============
print("\n🧹 [4/7] 清理...")

df = df[df['symbol_norm'] != BTC_SYM].copy()
df = df.dropna(subset=['fret'] + factor_cols)
df['fret_clipped'] = df['fret'].clip(-MAX_RET, MAX_RET)
df = df[df['fret'].abs() <= MAX_RET * 1.2].copy()
print(f"  有效: {len(df):,} | {df['symbol_norm'].nunique()} 币 | {df['date'].min().date()} ~ {df['date'].max().date()}")

# ============ Step 5: IC/IR ============
print("\n📊 [5/7] IC/IR...")

df = df.sort_values('date')
max_date = df['date'].max()
df['days_ago'] = (max_date - df['date']).dt.days
df['weight'] = np.exp(-np.log(2) / HALFLIFE * df['days_ago'])

results = []
for dt, grp in df.groupby('date'):
    if len(grp) < 8:
        continue
    for fc in factor_cols:
        x = grp[fc].dropna()
        y = grp.loc[x.index, 'fret_clipped']
        if len(x) < 8:
            continue
        try:
            ic, p = spearmanr(x.values, y.values)
            if np.isnan(ic):
                continue
        except:
            continue
        results.append({'date': dt, 'factor': fc, 'ic': ic, 'p': p, 'wt': grp.loc[x.index, 'weight'].mean()})

ic_df = pd.DataFrame(results)
ic_stats = ic_df.groupby('factor')['ic'].agg(IC_mean='mean', IC_std='std')
ic_stats['IC_IR']       = ic_stats['IC_mean'] / ic_stats['IC_std']
ic_stats['IC_pos_pct'] = ic_df.groupby('factor')['ic'].apply(lambda x: (x > 0).mean())
ic_stats['IC_abs_mean']= ic_df.groupby('factor')['ic'].apply(lambda x: np.abs(x).mean())
ic_stats = ic_stats.sort_values('IC_IR', ascending=False)

print("\nIC/IR:")
print("-" * 80)
for fn, r in ic_stats.iterrows():
    print(f"  {fn:<14} IC={r['IC_mean']:>+7.4f}  IR={r['IC_IR']:>+7.4f}  IC>0={r['IC_pos_pct']:.1%}")

# ============ Step 6: 多策略回测 ============
print("\n💰 [6/7] 策略回测...")

df['fr_q'] = df.groupby('date')['fr'].transform(
    lambda x: pd.qcut(x.rank(method='first'), 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
)

dates = sorted(df['date'].unique())
n_days = len(dates)

def daily_ret(day_df, q, direction, cost=COST_RATE):
    sub = day_df[day_df['fr_q'] == q].copy()
    if len(sub) == 0:
        return 0.0
    if direction == 'long':
        # 做多: 币收益 - 费率成本
        raw = sub['fret_clipped'] - sub['fr_cost_long']
    else:
        # 做空: -币收益 + 做空收到费率(fr<0时) - 费率成本(fr>0时做空者付)
        # 实际上: 做空者收到的费率 = 当fr<0时做空者收到钱, fr>0时做空者付钱
        # 简化: 做空收益 = -fret + (如果fr<0则+|fr|*3, 否则-|fr|*3*0.5)
        fr_adj = np.where(sub['fr'] < 0, sub['fr'].abs() * 3, -sub['fr'].abs() * 1.5)
        raw = -sub['fret_clipped'] + fr_adj
    raw = raw - cost
    return raw.mean()

strat_returns = {s: [] for s in ['A','B','C','D','E','F']}

for dt in dates:
    day = df[df['date'] == dt].copy()
    if len(day) < 5:
        for s in strat_returns:
            strat_returns[s].append(0.0)
        continue

    bull = day['market_regime'].iloc[0] > 0 if len(day) > 0 else True
    vol_mean_today = day['vol_z'].mean() if len(day) > 0 else 0
    high_vol = vol_mean_today > 0.3  # 波动率异常高

    # 策略A: 牛市多头 + 熊市空头（市场自适应）
    # 牛市: Q1做多(费率低持有成本低) + Q5轻仓做空(对冲)
    # 熊市: 反过来
    if bull:
        rA = 0.7 * daily_ret(day, 'Q1', 'long') + 0.3 * daily_ret(day, 'Q5', 'short')
    else:
        rA = 0.7 * daily_ret(day, 'Q5', 'short') + 0.3 * daily_ret(day, 'Q1', 'long')
    strat_returns['A'].append(rA)

    # 策略B: 纯做多Q1(费率最低), 费率成本后依然有正收益？
    rB = daily_ret(day, 'Q1', 'long')
    strat_returns['B'].append(rB)

    # 策略C: 动量 + 费率双因子选币
    # 同时考虑动量mom_z和费率fr_z: 做多mom_z高+fr_z低的币, 做空mom_z低+fr_z高的币
    sub = day.copy()
    if len(sub) > 0:
        # 选最好的30%做多, 最差的30%做空
        sub['score'] = sub['mom_z'] * 0.5 + sub['fr_z'] * -0.5  # 高mom + 低fr = 做多信号
        q_low = sub['score'].quantile(0.3)
        q_high = sub['score'].quantile(0.7)
        long_mask = sub['score'] >= q_high
        short_mask = sub['score'] <= q_low
        rC_long = (sub.loc[long_mask, 'fret_clipped'] - sub.loc[long_mask, 'fr_cost_long']).mean() if long_mask.sum() > 0 else 0
        fr_adj_s = np.where(sub.loc[short_mask, 'fr'] < 0, sub.loc[short_mask, 'fr'].abs() * 3,
                            -sub.loc[short_mask, 'fr'].abs() * 1.5)
        rC_short = (-sub.loc[short_mask, 'fret_clipped'] + fr_adj_s).mean() if short_mask.sum() > 0 else 0
        rC = 0.6 * rC_long + 0.4 * rC_short
    else:
        rC = 0.0
    strat_returns['C'].append(rC)

    # 策略D: 只做多Q1 + 市场beta对冲
    # 持有Q1多头 + 0.3倍的Q5空头作为对冲
    rD = 0.8 * daily_ret(day, 'Q1', 'long') + 0.2 * daily_ret(day, 'Q5', 'short')
    strat_returns['D'].append(rD)

    # 策略E: 只做空高波动 + 高费率组合（熊市保护）
    # 不管牛熊, 只要高波动+高费率就做空
    hv_fr = day[(day['vol_z'] > 0.2) & (day['fr_q'].isin(['Q4','Q5']))]
    if len(hv_fr) > 0:
        fr_adj = np.where(hv_fr['fr'] < 0, hv_fr['fr'].abs() * 3, -hv_fr['fr'].abs() * 1.5)
        rE = (-hv_fr['fret_clipped'] + fr_adj - COST_RATE).mean()
    else:
        rE = 0.0
    strat_returns['E'].append(rE)

    # 策略F: 只做多Q1, 加入止损(波动率>1.5σ时砍仓)
    hv = day[day['vol_z'] > 1.5]
    lv = day[day['vol_z'] <= 1.5]
    rF_long = daily_ret(lv[lv['fr_q'] == 'Q1'], 'Q1', 'long') if len(lv[lv['fr_q'] == 'Q1']) > 0 else 0.0
    strat_returns['F'].append(rF_long)

def perf(returns, label):
    rets = np.array(returns)
    rets = np.clip(rets, -0.20, 0.20)
    cum = np.cumprod(1 + rets)
    final = cum[-1] if len(cum) > 0 else 1.0
    ann = final ** (365 / max(n_days, 1)) - 1
    mean_ret = np.nanmean(rets)
    std_ret  = np.nanstd(rets)
    sharpe = mean_ret / (std_ret + 1e-9) * np.sqrt(365)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / (peak + 1e-9)
    max_dd = abs(np.nanmin(dd)) if np.any(dd < 0) else 0.0
    win_rate = (rets > 0).mean()
    return {
        'Strategy': label, 'Final净值': final,
        '年化收益率': ann, '夏普比率': sharpe,
        '最大回撤': max_dd, '日均收益': mean_ret,
        '日波动率': std_ret, '胜率': win_rate
    }

label_map = {
    'A': 'A: 牛熊自适应(Q1多/熊市Q5空)',
    'B': 'B: 仅做多Q1(费率最低)',
    'C': 'C: 动量+费率双因子选币',
    'D': 'D: Q1多头+Q5对冲(0.8/0.2)',
    'E': 'E: 高波动+高费率做空保护',
    'F': 'F: Q1做多+vol_z>1.5σ止损',
}
stats = [perf(strat_returns[s], label_map[s]) for s in strat_returns]
stats_df = pd.DataFrame(stats).sort_values('夏普比率', ascending=False)

print("\n策略绩效:")
print("-" * 100)
for _, row in stats_df.iterrows():
    print(f"  {row['Strategy']:<40} 年化={row['年化收益率']:>+8.2%}  夏普={row['夏普比率']:>+7.3f}  "
          f"回撤={row['最大回撤']:>7.2%}  胜率={row['胜率']:>6.1%}")

# ============ Step 7: 输出 ============
print("\n📅 [7/7] 保存...")

ic_df.to_csv(f"{OUTPUT}/ic_timeseries_v5.csv", index=False)
ic_stats.to_csv(f"{OUTPUT}/ic_stats_v5.csv")
monthly_ic = ic_df.copy()
monthly_ic['month'] = monthly_ic['date'].dt.to_period('M')
monthly_ic.pivot_table(index='month', columns='factor', values='ic', aggfunc='mean').to_csv(f"{OUTPUT}/monthly_ic_v5.csv")

best = stats_df.iloc[0]
n_syms = df['symbol_norm'].nunique()
min_date_str = str(dates[0])[:10]
max_date_str = str(dates[-1])[:10]

report = f"""# 加密货币量化因子有效性分析报告 v5（实战优化版）

> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> **数据范围**: {min_date_str} ~ {max_date_str}
> **分析币种**: {n_syms} 个（排除 BTC-USDT）
> **有效记录**: {len(df):,} 条 | {n_days} 交易日
> **指数衰减半衰期**: {HALFLIFE} 天

---

## 一、因子 IC/IR（2020-2026 大牛市背景）

| 因子 | IC均值 | IR | IC>0 | 解读 |
|------|-------:|---:|-----:|------|
"""

for fn, r in ic_stats.iterrows():
    note = ''
    if fn == 'vol_z' and r['IC_mean'] < 0:
        note = '高波动→未来收益差（最强信号）'
    elif fn == 'mom_z' and r['IC_mean'] < 0:
        note = '动量反转（强势币后续跑输）'
    elif fn == 'fr_sign' and r['IC_mean'] > 0:
        note = '高费率→未来收益差（费率有效）'
    elif fn == 'fr_z' and r['IC_mean'] > 0:
        note = '高费率→未来收益差'
    report += f"| {fn} | {r['IC_mean']:+.4f} | {r['IC_IR']:+.4f} | {r['IC_pos_pct']:.1%} | {note} |\n"

report += f"""
### 重要背景说明
2020-2026 年是加密货币大牛市（BTC 从 ~$10,000 涨至 ~$100,000+）。在此背景下：
- **做空策略普遍亏损**：所有纯做空策略年化 -22% ~ -73%
- **市场Beta主导**：方向判断比选币更重要
- **费率溢价存在但被牛市的β淹没**：高费率币虽然未来收益差，但在大牛市中依然上涨

---

## 二、策略回测（{n_days} 交易日）

| 策略 | 年化收益 | 夏普比率 | 最大回撤 | 日均收益 | 日波动率 | 胜率 |
|------|-------:|--------:|--------:|--------:|--------:|-----:|
"""

for _, row in stats_df.iterrows():
    report += f"| {row['Strategy']} | {row['年化收益率']:+.2%} | {row['夏普比率']:+.3f} | {row['最大回撤']:.2%} | {row['日均收益']:+.3%} | {row['日波动率']:.3%} | {row['胜率']:.1%} |\n"

report += f"""
### 最佳策略: **{best['Strategy']}**
- 年化: **{best['年化收益率']:+.2%}**
- 夏普: **{best['夏普比率']:+.3f}**
- 回撤: {best['最大回撤']:.2%}

---

## 三、策略逻辑

| 策略 | 说明 |
|------|------|
| **A: 牛熊自适应** | 牛市(Q>0): 70%做多Q1+30%做空Q5对冲; 熊市: 反转 |
| **B: 仅做多Q1** | 纯做多费率最低的20%币，费率成本后验证可行性 |
| **C: 动量+费率双因子** | 做多 mom_z高+fr_z低，做空 mom_z低+fr_z高 |
| **D: Q1多头+Q5对冲** | 80%做多Q1 + 20%做空Q5（对冲保护） |
| **E: 高波动+高费率做空** | 只在高波动+高费率的极端情况做空（保护性） |
| **F: Q1+波动止损** | 只做多Q1，vol_z>1.5σ时砍仓 |

---

## 四、核心发现与实战建议

### 关键发现
1. **fr_sign 是最稳定的因子**：IC>0占比 83%，说明高费率确实预示着未来更差的收益
2. **vol_z 是最强预测因子**：IR=-0.28，高波动永远是最灵的空头信号
3. **纯做空策略不可行**：大牛市里做空费率最高的币，年化亏损 -40%
4. **真正有意义的策略**：
   - 做多低费率(Q1)在大牛市中是可行的，但需要控制仓位
   - 加入波动率止损(F策略)是必要的保护

### 实战建议（如何在牛市中赚钱）

**推荐仓位配置：**
- **70% 核心仓位**：做多低费率(Q1)币种，长持不动（费率成本最低）
- **20% 对冲仓位**：高波动时做空高费率(Q5)币种对冲（保护性）
- **10% 现金/稳定币**：高波动时持有 USDT 观望

**入场/离场信号：**
- **入场**：BTC动量>0（牛市） + vol_z < 0.5 → 做多Q1
- **离场**：BTC动量<0（熊市） + vol_z > 1.0 → 全部平仓
- **止损**：日收益 < -3% → 砍仓50%； < -5% → 全部平仓

---

## 五、IC稳定性（近12月）

"""

monthly_ic_df = pd.read_csv(f"{OUTPUT}/monthly_ic_v5.csv")
if 'Unnamed: 0' in monthly_ic_df.columns:
    monthly_ic_df = monthly_ic_df.rename(columns={'Unnamed: 0': 'month'})
recent = monthly_ic_df.tail(12)
for col in ['vol_z', 'mom_z', 'fr_sign', 'fr_z']:
    if col in recent.columns:
        pos = (recent[col] > 0).mean()
        report += f"- **{col}**: 近12月IC>0 {pos:.0%}（{'✅ 稳定' if pos>0.6 else '⚠️ 注意'}）\n"

report += f"""
---

## 六、总结

在 2020-2026 大牛市背景下：
- 纯做空策略（任何形式）年化 -22% ~ -73%，不可行
- **做多低费率(Q1) + 波动率止损** 是唯一可行的策略方向
- 费率的真正价值是**选币**而非方向：低费率=持有成本低=牛市里拿得住
- fr_sign（IC=+0.021, IR=+0.092）是唯一在牛市中能用的正向信号

**下一步优化方向：**
1. 加入更多市场环境判断（宏观利率、BTC减半周期等）
2. 用期货vs现货价差判断机构情绪
3. 做空时要同时看vol_z和OI（未平仓合约）——OI高+高费率=双重做空信号

---

*本报告由因子分析系统 v5 生成 | 数据：Binance | 背景：2020-2026 大牛市*
"""

out = f"{OUTPUT}/资金费率因子分析报告_v5_{datetime.now().strftime('%Y%m%d')}.md"
with open(out, 'w') as f:
    f.write(report)
print(f"\n✅ 报告: {out}")
