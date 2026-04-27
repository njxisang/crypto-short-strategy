# 数据目录

本项目使用 Binance 永续合约历史数据，数据文件较大（>1.8GB），存储于父目录：

```
/home/xisang/crypto-new/binance_data/
```

## 数据文件

| 目录 | 说明 | 记录频率 |
|------|------|---------|
| `binance-swap-candle-csv-1h/` | 永续合约1小时K线 | 1小时 |
| `binance_funding_rate/usdt/` | 资金费率（8小时一次） | 8小时 |
| `coin-cap/` | 链上流通量（CoinCap） | 日 |

## 数据来源

- K线 & 资金费率：Binance API 公开历史数据
- 链上数据：[CoinCap API](https://coincap.io/)

## 数据字段

### 永续合约K线 (`*USDT.csv`)

```
candle_begin_time, open, high, low, close, volume, quote_volume,
trade_num, taker_buy_base_asset_volume, taker_buy_quote_asset_volume,
Spread, symbol, avg_price_1m, avg_price_5m, fundingRate
```

### 资金费率 (`usdt/*USDT.csv`)

```
time, symbol, fundingRate
```

### 链上流通量 (`coin-cap/*BTC.csv`)

```
candle_begin_time, symbol, circulating_supply
```
