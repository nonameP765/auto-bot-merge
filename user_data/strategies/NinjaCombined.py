"""
NinjaCombined v1.0 - bidirectional 5m futures strategy (long + short).

Combines NinjaFutures5m V14.0 (long side) and NinjaForgeShort v16.0 (short side)
into one strategy with fully separated per-side entry, exit, and stoploss behavior.

Protection architecture note:
- Built-in StoplossGuard is intentionally NOT used because built-in protections do not
  distinguish long vs short.
- StoplossGuard is implemented manually per side in confirm_trade_entry().
- Only CooldownPeriod remains as a built-in shared protection.

Long side (NinjaFutures5m V14.0):
- Entry signals: e0v1e_1, e0v1e_new, clucHA, cofi.
- Exit model: custom_stoploss with long-only trailing + supplemental custom exits.
- Base stoploss: -20% (via custom_stoploss).

Short side (NinjaForgeShort v16.0):
- Entry signals: bear_bounce_short, correction_fade_short.
- Exit model: R-multiple priority stack with winner-exit deferral wrapper.
- Base stoploss: -35% (fixed, no trailing).

Technical merge decisions:
- Global stoploss = -0.35 to preserve short-side risk model; long side narrows to -0.20
  and applies trailing in custom_stoploss.
- Global trailing_stop = False; trailing is implemented manually for longs only.
- minimal_roi = {"360": 0.04} retained as long-side safety fallback while custom exits
  handle primary exit flow.
- Protections are split by side in confirm_trade_entry(); built-in protections keep only
  CooldownPeriod.

Execution profile:
- Timeframes: 5m main + 1h informative regime context.
- Leverage: 15x for both long and short.
- Exposure: max 1 concurrent position.

Latest backtest results (start $1,000, 75% wallet, compound, protections enabled,
cache none):
- P1 (2020.07-2022.05): 3,757,869% | 617 trades | WR 68.2% | DD 71.73%
- P2 (2022.05-2024.03): 17,937% | 173 trades | WR 60.7% | DD 64.67%
- P3 (2024.03-2025.12): 149,417% | 175 trades | WR 72.0% | DD 28.85%
- OOS (2026.01-2026.03): 109.75% | 14 trades | WR 78.6% | DD 26.21%
"""

from datetime import timedelta, datetime
from typing import Optional

import numpy as np
import pandas_ta as pta
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_from_open


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Correlation Trend Indicator (from NinjaFutures5m)
# ─────────────────────────────────────────────────────────────────────────────
def CTI(close, length=20):
    """Pearson correlation between price and linear trend. +1=uptrend, -1=downtrend."""
    x = np.arange(length)
    return close.rolling(length).apply(
        lambda y: np.corrcoef(x, y)[0, 1] if len(y) == length else 0,
        raw=True,
    )


class NinjaCombined(IStrategy):
    INTERFACE_VERSION: int = 3

    can_short: bool = True
    timeframe = "5m"
    startup_candle_count = 2494  # Short needs deep warm-up for EMA200 1h
    process_only_new_candles = True

    # ═════════════════════════════════════════════════════════════════════════
    # STOPLOSS / TRAILING / ROI
    # ═════════════════════════════════════════════════════════════════════════
    stoploss = -0.35  # Widest base (short). custom_stoploss narrows for long.
    use_custom_stoploss = True
    trailing_stop = False  # Trailing implemented per-side in custom_stoploss

    minimal_roi = {
        "360": 0.04
    }  # Long's safety exit. Shorts exit via custom_exit first.

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    position_adjustment_enable = False

    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # ═════════════════════════════════════════════════════════════════════════
    # SHORT PARAMETERS — verbatim from NinjaForgeShort
    # ═════════════════════════════════════════════════════════════════════════

    # Regime detection (1h)
    REGIME_SLOPE_PERIODS = 6
    BULL_ADX_MIN = 24
    BULL_CHOP_MAX = 50
    BULL_SLOPE_THRESHOLD = 0.002
    BULL_DEBOUNCE = 3
    BULL_EMA200_MARGIN = 1.005
    BEAR_ADX_MIN = 23
    BEAR_CHOP_MAX = 50
    BEAR_SLOPE_THRESHOLD = 0.0035
    BEAR_DEBOUNCE = 3
    RANGE_ADX_MAX = 18
    RANGE_CHOP_MIN = 56
    RANGE_SLOPE_ABS_MAX = 0.0012
    RANGE_DEBOUNCE = 3
    BULL_ACCEL_ADX_MIN = 20
    BULL_ACCEL_CHOP_MAX = 55
    BULL_ACCEL_EMA50_SLOPE_PERIODS = 4
    ST_PERIOD = 10
    ST_MULTIPLIER = 3.0
    TRANSITION_LOOKBACK = 168
    RECOVERY_BEAR_STREAK_MIN = 4
    RECOVERY_EMA50_LOOKBACK = 24
    RECOVERY_SMA200_SLOPE_THRESHOLD = -0.15
    RECOVERY_EMA50_RISING_THRESHOLD = 0.005

    # Short entry / exit thresholds
    TREND_ADX_MIN = 22
    TREND_CHOP_MAX = 52
    TREND_VOL_MULT = 0.9
    TREND_STOP_PCT = 0.020
    TREND_BB_EXIT_R = 0.51
    TREND_EXHAUSTION_R = 0.43
    INVALIDATION_PROFIT_GATE = -0.25
    INVALIDATION_RSI_GATE_PROFIT = -0.15
    TREND_FADE_MIN_HOURS = 2.0
    TREND_FADE_ADX_MAX = 20

    SHORT_LEVERAGE = 15.0
    SHORT_RSI_MIN = 37
    SHORT_RSI_MAX = 62
    TREND_TP_SHORT_R = 2.3
    TREND_SHORT_MAX_HOLD = 7
    MOM_EXIT_SHORT = -0.28
    TREND_FADE_PROFIT_GATE = -0.25

    # Winner exit deferral
    DEFER_MIN_R = 1.0
    DEFER_ADX_MIN = 20
    DEFER_MAX_MINUTES = 40
    DEFER_CANCEL_R = 0.3
    DEFER_PROFIT_EXITS = {
        "bb_lower_exit",
        "stoch_exhaustion_exit",
        "willr_exhaustion_exit",
        "macd_exhaustion_exit",
        "stagnation_exit",
        "chop_stagnation_exit",
        "time_exit_profit",
    }

    # Correction fade short
    CORR_RSI_MIN = 35
    CORR_RSI_MAX = 56
    CORR_EWO_THRESHOLD = -0.5
    CORR_ADX_MIN = 23
    CORR_CHOP_MAX = 52
    CORR_ATR_MULT = 1.3
    CORR_TP_SHORT_R = 1.5
    CORR_MAX_HOLD = 5
    CORR_STAGNATION_HOURS = 3

    POST_SL_GATE_HOURS = 10

    # ═════════════════════════════════════════════════════════════════════════
    # LONG PARAMETERS — verbatim from NinjaFutures5m
    # ═════════════════════════════════════════════════════════════════════════

    LONG_LEVERAGE = 15
    LONG_STOPLOSS = -0.20
    LONG_TRAILING_OFFSET = (
        0.04  # trailing_stop_positive_offset (V14.0: was 0.05 in V13.0)
    )
    LONG_TRAILING_STOP = 0.02  # trailing_stop_positive

    # Entry (e0v1e signals)
    buy_rsi_fast_limit = 40
    buy_rsi_limit = 42
    buy_sma15_ratio = 0.973
    buy_cti_limit = 0.69
    buy_new_rsi_fast = 34
    buy_new_rsi = 28
    buy_new_sma15_ratio = 0.96

    # Exit (supplementary)
    sell_fastx = 90
    sell_cci_threshold = 120
    sell_cci_profit = -0.05

    # ═════════════════════════════════════════════════════════════════════════
    # PER-SIDE PROTECTION PARAMETERS
    # ═════════════════════════════════════════════════════════════════════════
    #
    # Built-in protections can't distinguish long vs short.
    # StoplossGuard is implemented per-side in confirm_trade_entry instead.
    # Only CooldownPeriod remains built-in (shared, same value for both).
    #
    # Long protections (from NinjaFutures5m V14.0):
    #   StoplossGuard: 1 long SL in 4h → block long 6h (all pairs)
    LONG_SL_GUARD_LOOKBACK_H = 4
    LONG_SL_GUARD_TRADE_LIMIT = 1
    LONG_SL_GUARD_PAUSE_H = 6

    # Short protections (from NinjaForgeShort v16.0):
    #   Tier 1: 2 short SLs on same pair in 2h → block that pair 2h
    SHORT_SL_GUARD_PAIR_LOOKBACK_H = 2
    SHORT_SL_GUARD_PAIR_TRADE_LIMIT = 2
    SHORT_SL_GUARD_PAIR_PAUSE_H = 2
    #   Tier 2: 2 short SLs in 10h → block all shorts 12h
    SHORT_SL_GUARD_CASCADE_LOOKBACK_H = 10
    SHORT_SL_GUARD_CASCADE_TRADE_LIMIT = 2
    SHORT_SL_GUARD_CASCADE_PAUSE_H = 12
    #   Tier 3: 3 short SLs in 24h → block all shorts 18h
    SHORT_SL_GUARD_PORTFOLIO_LOOKBACK_H = 24
    SHORT_SL_GUARD_PORTFOLIO_TRADE_LIMIT = 3
    SHORT_SL_GUARD_PORTFOLIO_PAUSE_H = 18

    # ═════════════════════════════════════════════════════════════════════════
    # PROTECTIONS — CooldownPeriod only (StoplossGuard moved to per-side)
    # ═════════════════════════════════════════════════════════════════════════
    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 3},
        ]

    def informative_pairs(self):
        return [(pair, "1h") for pair in self.dp.current_whitelist()]

    # ═════════════════════════════════════════════════════════════════════════
    # INDICATORS — both 5m stacks + shared 1h informative
    # ═════════════════════════════════════════════════════════════════════════
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Compute merged indicator stacks: long-side 5m, short-side 5m, and shared 1h regime context."""
        # ── 5m indicators shared by both sides ──
        dataframe["ema_8"] = ta.EMA(dataframe, timeperiod=8)
        stoch_fast = ta.STOCHF(
            dataframe, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        dataframe["fastk"] = stoch_fast["fastk"]
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        # ── 5m indicators for SHORT (NinjaForgeBase) ──
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bollinger["upperband"]
        dataframe["bb_lower"] = bollinger["lowerband"]
        dataframe["willr"] = ta.WILLR(dataframe, timeperiod=14)
        atr_sum = dataframe["atr"].rolling(14).sum()
        high_14 = dataframe["high"].rolling(14).max()
        low_14 = dataframe["low"].rolling(14).min()
        hl_diff = (high_14 - low_14).replace(0, np.nan)
        dataframe["chop"] = (100 * np.log10(atr_sum / hl_diff) / np.log10(14)).fillna(
            50
        )
        dataframe["atr_sma"] = dataframe["atr"].rolling(window=50).mean()
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()
        candle_range = (dataframe["high"] - dataframe["low"]).replace(0, np.nan)
        dataframe["body_ratio"] = (
            (dataframe["close"] - dataframe["open"]).abs() / candle_range
        ).fillna(0)
        ema_fast_5 = ta.EMA(dataframe, timeperiod=5)
        ema_slow_35 = ta.EMA(dataframe, timeperiod=35)
        close_safe = dataframe["close"].replace(0, np.nan)
        dataframe["ewo"] = ((ema_fast_5 - ema_slow_35) / close_safe * 100).fillna(0)

        # ── 5m indicators for LONG (NinjaFutures5m) ──
        dataframe["sma_15"] = ta.SMA(dataframe, timeperiod=15)
        dataframe["cti"] = CTI(dataframe["close"], length=20)
        dataframe["rsi_20"] = ta.RSI(dataframe, timeperiod=20)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=40)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["ma120"] = ta.MA(dataframe, timeperiod=120)

        # Heikin-Ashi + BB(40) for clucHA signal
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["ha_close"] = heikinashi["close"]
        dataframe["ha_open"] = heikinashi["open"]
        dataframe["ha_high"] = heikinashi["high"]
        dataframe["ha_low"] = heikinashi["low"]
        ha_tp = (dataframe["ha_high"] + dataframe["ha_low"] + dataframe["ha_close"]) / 3
        ha_bb_mid = ha_tp.rolling(40).mean()
        ha_bb_std = ha_tp.rolling(40).std()
        dataframe["bb_lowerband2_40"] = ha_bb_mid - 2 * ha_bb_std
        dataframe["bb_delta_cluc"] = (ha_bb_mid - dataframe["bb_lowerband2_40"]).abs()
        dataframe["ha_closedelta"] = (
            dataframe["ha_close"] - dataframe["ha_close"].shift()
        ).abs()
        dataframe["tail"] = (dataframe["ha_close"] - dataframe["ha_low"]).abs()
        dataframe["rocr_28"] = ta.ROCR(dataframe["ha_close"], timeperiod=28)

        # ── 1h informative ──
        informative = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")
        if len(informative) > 0:
            # --- Shared 1h indicators ---
            informative["ema_200"] = ta.EMA(informative, timeperiod=200)
            informative["ema_50"] = ta.EMA(informative, timeperiod=50)
            informative["sma_200"] = ta.SMA(informative, timeperiod=200)
            informative["rsi_14"] = ta.RSI(informative, timeperiod=14)
            informative["rsi_14_slope"] = informative["rsi_14"] - informative[
                "rsi_14"
            ].shift(1)
            informative["adx"] = ta.ADX(informative, timeperiod=14)
            informative["atr"] = ta.ATR(informative, timeperiod=14)

            # EMA200 slope (regime detection)
            sp = self.REGIME_SLOPE_PERIODS
            informative["ema200_slope"] = (
                informative["ema_200"] - informative["ema_200"].shift(sp)
            ) / informative["ema_200"].shift(sp)

            # Chop index 1h
            atr_sum_1h = informative["atr"].rolling(14).sum()
            high_14_1h = informative["high"].rolling(14).max()
            low_14_1h = informative["low"].rolling(14).min()
            hl_diff_1h = (high_14_1h - low_14_1h).replace(0, np.nan)
            informative["chop"] = (
                100 * np.log10(atr_sum_1h / hl_diff_1h) / np.log10(14)
            ).fillna(50)

            # EMA50 slope for BULL_ACCEL
            ba_sp = self.BULL_ACCEL_EMA50_SLOPE_PERIODS
            ema50_shifted = informative["ema_50"].shift(ba_sp)
            informative["ema50_slope"] = (
                (informative["ema_50"] - ema50_shifted) / ema50_shifted
            ).fillna(0)

            # EMA50 slope 24h for recovery detection
            rec_sp = self.RECOVERY_EMA50_LOOKBACK
            ema50_shifted_24 = informative["ema_50"].shift(rec_sp)
            informative["ema50_slope_24h"] = (
                (informative["ema_50"] - ema50_shifted_24) / ema50_shifted_24
            ).fillna(0)

            # SMA200 slope for recovery detection
            sma200_prev_20 = informative["sma_200"].shift(20)
            informative["sma200_slope_pct"] = (
                (informative["sma_200"] - sma200_prev_20) / sma200_prev_20 * 100
            ).fillna(0)

            # SuperTrend on 1h
            st_1h = pta.supertrend(
                informative["high"],
                informative["low"],
                informative["close"],
                length=self.ST_PERIOD,
                multiplier=self.ST_MULTIPLIER,
            )
            st_col = f"SUPERTd_{self.ST_PERIOD}_{self.ST_MULTIPLIER}"
            informative["st_direction"] = st_1h[st_col]

            # --- Long-specific 1h indicators ---
            informative["roc_9"] = ta.ROC(informative, timeperiod=9)
            bb_1h = ta.BBANDS(informative, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
            informative["bb_width"] = (bb_1h["upperband"] - bb_1h["lowerband"]) / bb_1h[
                "middleband"
            ]

            # ── REGIME DETECTION (verbatim from NinjaForgeBase) ──

            # BULL regime
            bull_raw = (
                (
                    informative["close"]
                    > informative["ema_200"] * self.BULL_EMA200_MARGIN
                )
                & (informative["ema_50"] > informative["ema_200"])
                & (informative["ema200_slope"] > self.BULL_SLOPE_THRESHOLD)
                & (informative["adx"] >= self.BULL_ADX_MIN)
                & (informative["chop"] <= self.BULL_CHOP_MAX)
            )
            bull_debounced = bull_raw.copy()
            for i in range(1, self.BULL_DEBOUNCE):
                bull_debounced = bull_debounced & bull_raw.shift(i, fill_value=False)
            informative["bull_regime"] = bull_debounced.astype(int)

            # BEAR regime
            bear_raw = (
                (informative["close"] < informative["ema_200"])
                & (informative["close"] < informative["ema_50"])
                & (informative["ema_50"] < informative["ema_200"])
                & (informative["ema200_slope"] < -self.BEAR_SLOPE_THRESHOLD)
                & (informative["adx"] >= self.BEAR_ADX_MIN)
                & (informative["chop"] <= self.BEAR_CHOP_MAX)
            )
            if self.BEAR_DEBOUNCE <= 1:
                informative["bear_regime"] = bear_raw.astype(int)
            else:
                bear_debounced = bear_raw.copy()
                for i in range(1, self.BEAR_DEBOUNCE):
                    bear_debounced = bear_debounced & bear_raw.shift(
                        i, fill_value=False
                    )
                informative["bear_regime"] = bear_debounced.astype(int)

            # RANGE regime
            range_raw = (
                (informative["adx"] <= self.RANGE_ADX_MAX)
                & (informative["chop"] >= self.RANGE_CHOP_MIN)
                & (informative["ema200_slope"].abs() <= self.RANGE_SLOPE_ABS_MAX)
            )
            range_debounced = range_raw.copy()
            for i in range(1, self.RANGE_DEBOUNCE):
                range_debounced = range_debounced & range_raw.shift(i, fill_value=False)
            informative["range_regime"] = (
                range_debounced
                & (informative["bull_regime"] == 0)
                & (informative["bear_regime"] == 0)
            ).astype(int)

            # CORRECTION regime (catch-all)
            informative["correction_regime"] = (
                (informative["bull_regime"] == 0)
                & (informative["bear_regime"] == 0)
                & (informative["range_regime"] == 0)
            ).astype(int)

            # BULL_ACCEL overlay
            bull_accel_raw = (
                (informative["close"] > informative["ema_50"])
                & (informative["ema50_slope"] > 0)
                & (informative["adx"] >= self.BULL_ACCEL_ADX_MIN)
                & (informative["chop"] <= self.BULL_ACCEL_CHOP_MAX)
                & (informative["bear_regime"] == 0)
            )
            informative["bull_accel_regime"] = (
                bull_accel_raw & (informative["bull_regime"] == 0)
            ).astype(int)

            # TRANSITION regime
            bull_any_1h = (informative["bull_regime"] == 1) | (
                informative["bull_accel_regime"] == 1
            )
            bull_was_recent = (
                bull_any_1h.rolling(self.TRANSITION_LOOKBACK, min_periods=1)
                .max()
                .fillna(0)
            )
            informative["transition_regime"] = (
                (informative["bear_regime"] == 1) & (bull_was_recent == 1)
            ).astype(int)

            # Bear streak
            bear_int = informative["bear_regime"]
            bear_groups = (bear_int != bear_int.shift()).cumsum()
            informative["bear_streak"] = bear_int.groupby(bear_groups).cumcount() + 1
            informative["bear_streak"] = informative["bear_streak"].where(
                bear_int == 1, 0
            )

            # Correction age + micro-subset
            corr_int = informative["correction_regime"]
            corr_groups = (corr_int != corr_int.shift()).cumsum()
            informative["correction_age"] = corr_int.groupby(corr_groups).cumcount() + 1
            informative["correction_age"] = informative["correction_age"].where(
                corr_int == 1, 0
            )
            corr_micro_age1 = (informative["correction_age"] == 1) & (
                informative["bear_streak"].shift(1) >= 10
            )
            corr_micro_age2 = (informative["correction_age"] == 2) & (
                informative["bear_streak"].shift(2) >= 10
            )
            informative["corr_micro_ok"] = (
                (corr_micro_age1 | corr_micro_age2) & (informative["adx"] >= 30)
            ).astype(int)

            # RECOVERY regime
            recovery_path1 = (
                informative["sma200_slope_pct"] > self.RECOVERY_SMA200_SLOPE_THRESHOLD
            ) & (informative["sma200_slope_pct"] < 0)
            recovery_path2 = (
                informative["ema50_slope_24h"] > self.RECOVERY_EMA50_RISING_THRESHOLD
            )
            informative["recovery_regime"] = (
                (informative["bear_regime"] == 1) & (recovery_path1 | recovery_path2)
            ).astype(int)

            # Hours since last BEAR
            bear_bool = informative["bear_regime"] == 1
            informative["hours_since_bear"] = (
                (~bear_bool)
                .groupby((bear_bool != bear_bool.shift()).cumsum())
                .cumcount()
            )
            informative["hours_since_bear"] = informative["hours_since_bear"].where(
                bear_bool.cumsum() > 0, 9999
            )

            # WEAK_BEAR overlay
            informative["weak_bear_regime"] = (
                (informative["correction_regime"] == 1)
                & (informative["st_direction"] == -1)
                & (informative["ema200_slope"] < -0.002)
                & (informative["close"] < informative["ema_200"])
                & (informative["ema_50"] < informative["ema_200"])
                & (informative["close"] < informative["ema_50"])
                & (informative["adx"] >= 18)
                & (informative["hours_since_bear"] <= 168)
            ).astype(int)

            # Merge 1h → 5m (adds _1h suffix to all informative columns)
            dataframe = merge_informative_pair(
                dataframe, informative, self.timeframe, "1h", ffill=True
            )
        else:
            # Fallback when no informative data
            dataframe["close_1h"] = dataframe["close"]
            dataframe["ema_200_1h"] = dataframe["close"]
            dataframe["ema_50_1h"] = dataframe["close"]
            dataframe["sma_200_1h"] = dataframe["close"]
            dataframe["rsi_14_1h"] = 50.0
            dataframe["rsi_14_slope_1h"] = 0.0
            dataframe["adx_1h"] = 25.0
            dataframe["ema200_slope_1h"] = 0.0
            dataframe["atr_1h"] = dataframe["atr"]
            dataframe["chop_1h"] = 50.0
            dataframe["ema50_slope_1h"] = 0.0
            dataframe["ema50_slope_24h_1h"] = 0.0
            dataframe["sma200_slope_pct_1h"] = 0.0
            dataframe["st_direction_1h"] = 0
            dataframe["roc_9_1h"] = 0.0
            dataframe["bb_width_1h"] = 0.5
            dataframe["bull_regime_1h"] = 0
            dataframe["bear_regime_1h"] = 0
            dataframe["range_regime_1h"] = 0
            dataframe["correction_regime_1h"] = 1
            dataframe["bull_accel_regime_1h"] = 0
            dataframe["transition_regime_1h"] = 0
            dataframe["corr_micro_ok_1h"] = 0
            dataframe["recovery_regime_1h"] = 0
            dataframe["bear_streak_1h"] = 0
            dataframe["weak_bear_regime_1h"] = 0
            dataframe["hours_since_bear_1h"] = 9999

        return dataframe

    # ═════════════════════════════════════════════════════════════════════════
    # ENTRY — both long and short signals
    # ═════════════════════════════════════════════════════════════════════════
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Build entry flags for 4 long dip signals and 2 short rejection/fade signals with regime gates."""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        # ─────────────────────────────────────────────────────────────────
        # LONG ENTRIES (from NinjaFutures5m)
        # ─────────────────────────────────────────────────────────────────

        # Gate A: BULL — price above rising EMA-200 on 1h
        gate_bull = (
            (dataframe["close_1h"] > dataframe["ema_200_1h"])
            & (dataframe["ema_200_1h"] > dataframe["ema_200_1h"].shift(12))
            & (dataframe["roc_9_1h"] < 120)
            & (dataframe["bb_width_1h"] < 0.95)
            & (dataframe["rsi_14_1h"] < 80)
            & (dataframe["volume"] > 0)
        )

        # Gate B: NEUTRAL — price above EMA-50 * 0.98 on 1h
        gate_neutral = (
            (dataframe["close_1h"] > dataframe["ema_50_1h"] * 0.98)
            & (dataframe["rsi_14_1h"] > 18)
            & (dataframe["rsi_14_1h"] < 78)
            & (dataframe["bb_width_1h"] < 1.0)
            & (dataframe["roc_9_1h"] > -8)
            & (dataframe["volume"] > 0)
        )

        # Entry 1: e0v1e_1 — Shallow RSI dip (BULL gate)
        e0v1e_1 = gate_bull & (
            (dataframe["rsi_slow"] < dataframe["rsi_slow"].shift(1))
            & (dataframe["rsi_fast"] < self.buy_rsi_fast_limit)
            & (dataframe["rsi_20"] > self.buy_rsi_limit)
            & (dataframe["close"] < dataframe["sma_15"] * self.buy_sma15_ratio)
            & (dataframe["cti"] < self.buy_cti_limit)
        )

        # Entry 2: e0v1e_new — Deep RSI dip (NEUTRAL gate)
        e0v1e_new = gate_neutral & (
            (dataframe["rsi_slow"] < dataframe["rsi_slow"].shift(1))
            & (dataframe["rsi_fast"] < self.buy_new_rsi_fast)
            & (dataframe["rsi_20"] > self.buy_new_rsi)
            & (dataframe["close"] < dataframe["sma_15"] * self.buy_new_sma15_ratio)
            & (dataframe["cti"] < self.buy_cti_limit)
        )

        # Gate C: NEUTRAL-STRICT — anti-knife filter for clucHA in non-bull regimes
        gate_neutral_strict = gate_neutral & (dataframe["rsi_14_1h"] > 25)

        # Entry 3: clucHA — Heikin-Ashi BB40 bounce (BULL or NEUTRAL-STRICT gate)
        clucHA_conditions = (
            (dataframe["rocr_28"] > 0.526)
            & (dataframe["bb_lowerband2_40"].shift() > 0)
            & (dataframe["bb_delta_cluc"] > dataframe["ha_close"] * 0.044)
            & (dataframe["ha_closedelta"] > dataframe["ha_close"] * 0.017)
            & (dataframe["tail"] < dataframe["bb_delta_cluc"] * 1.146)
            & (dataframe["ha_close"] < dataframe["bb_lowerband2_40"].shift())
            & (dataframe["ha_close"] < dataframe["ha_close"].shift())
        )
        clucHA = (gate_bull | gate_neutral_strict) & clucHA_conditions

        # Entry 4: cofi — Stochastic crossover in oversold (NEUTRAL gate)
        cofi_cross = (dataframe["fastk"] > dataframe["fastd"]) & (
            dataframe["fastk"].shift(1) <= dataframe["fastd"].shift(1)
        )
        cofi = gate_neutral & (
            (dataframe["bb_width_1h"] < 0.5)
            & (dataframe["close"] < dataframe["ema_8"] * 0.970)
            & cofi_cross
            & (dataframe["fastk"] < 20)
            & (dataframe["fastd"] < 20)
            & (dataframe["adx"] > 20)
            & (dataframe["cti"] < -0.5)
        )

        # Apply long entries (later overrides earlier on conflict)
        dataframe.loc[cofi, ["enter_long", "enter_tag"]] = (1, "cofi")
        dataframe.loc[clucHA, ["enter_long", "enter_tag"]] = (1, "clucHA")
        dataframe.loc[e0v1e_1, ["enter_long", "enter_tag"]] = (1, "e0v1e_1")
        dataframe.loc[e0v1e_new, ["enter_long", "enter_tag"]] = (1, "e0v1e_new")

        # ─────────────────────────────────────────────────────────────────
        # SHORT ENTRIES (from NinjaForgeShort)
        # ─────────────────────────────────────────────────────────────────

        bear_regime = dataframe["bear_regime_1h"] == 1
        weak_bear = dataframe["weak_bear_regime_1h"] == 1

        trend_filter = (dataframe["adx"] > self.TREND_ADX_MIN) & (
            dataframe["chop"] < self.TREND_CHOP_MAX
        )
        vol_ok = (
            dataframe["volume"] > dataframe["volume_sma"] * self.TREND_VOL_MULT
        ) & (dataframe["volume"] > 0)

        # Recovery context filter
        recovery_context = dataframe["ema50_slope_24h_1h"] > 0
        bear_streak_ok = dataframe["bear_streak_1h"] >= self.RECOVERY_BEAR_STREAK_MIN
        recovery_filter = ~recovery_context | bear_streak_ok

        # bear_bounce_short
        short_bounce = (
            bear_regime
            & (dataframe["st_direction_1h"] == -1)
            & (dataframe["close"] < dataframe["ema_21"])
            & (dataframe["high"] > dataframe["ema_21"])
            & (dataframe["close"] < dataframe["ema_50"])
            & (dataframe["close"].shift(1) > dataframe["close"])
            & (dataframe["rsi_14"] > self.SHORT_RSI_MIN)
            & (dataframe["rsi_14"] < self.SHORT_RSI_MAX)
            & (dataframe["rsi_14"] < dataframe["rsi_14"].shift(1))
            & (dataframe["willr"] > -85)
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["body_ratio"] > 0.3)
            & (dataframe["ema_8"] < dataframe["ema_21"])
            & trend_filter
            & (dataframe["atr"] < dataframe["atr_sma"] * 1.5)
            & recovery_filter
            & (dataframe["ewo"] < 0)
            & (dataframe["recovery_regime_1h"] == 0)
            & vol_ok
        )
        dataframe.loc[short_bounce, "enter_short"] = 1
        dataframe.loc[short_bounce, "enter_tag"] = "bear_bounce_short"

        # correction_fade_short
        candle_range_s = (dataframe["high"] - dataframe["low"]).replace(0, 1e-10)
        close_position = (dataframe["close"] - dataframe["low"]) / candle_range_s
        close_in_lower_half = close_position < 0.4

        correction_fade = (
            weak_bear
            & (dataframe["st_direction_1h"] == -1)
            & (dataframe["close"] < dataframe["ema_21"])
            & (dataframe["high"] > dataframe["ema_21"])
            & (dataframe["close"] < dataframe["ema_50"])
            & (dataframe["close"].shift(1) > dataframe["close"])
            & (dataframe["rsi_14"] > self.CORR_RSI_MIN)
            & (dataframe["rsi_14"] < self.CORR_RSI_MAX)
            & (dataframe["rsi_14"] < dataframe["rsi_14"].shift(1))
            & (dataframe["willr"] > -85)
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["body_ratio"] > 0.4)
            & close_in_lower_half
            & (dataframe["fastk"] < 70)
            & (dataframe["ema_8"] < dataframe["ema_21"])
            & (dataframe["adx"] > self.CORR_ADX_MIN)
            & (dataframe["chop"] < self.CORR_CHOP_MAX)
            & (dataframe["atr"] < dataframe["atr_sma"] * self.CORR_ATR_MULT)
            & (dataframe["ewo"] < self.CORR_EWO_THRESHOLD)
            & (dataframe["macd_hist"] < 0)
            & recovery_filter
            & (dataframe["recovery_regime_1h"] == 0)
            & vol_ok
            & ~short_bounce
        )
        dataframe.loc[correction_fade, "enter_short"] = 1
        dataframe.loc[correction_fade, "enter_tag"] = "correction_fade_short"

        return dataframe

    # ═════════════════════════════════════════════════════════════════════════
    # EXIT TREND — empty (exits handled by custom_exit / stoploss / trailing)
    # ═════════════════════════════════════════════════════════════════════════
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""
        return dataframe

    # ═════════════════════════════════════════════════════════════════════════
    # LEVERAGE — 15x for both sides
    # ═════════════════════════════════════════════════════════════════════════
    def leverage(
        self,
        pair,
        current_time,
        current_rate,
        proposed_leverage,
        max_leverage,
        entry_tag,
        side,
        **kwargs,
    ):
        """Use fixed 15x on both sides to keep merged risk and sizing behavior consistent."""
        return 15.0

    # ═════════════════════════════════════════════════════════════════════════
    # CUSTOM STOPLOSS — per-side stoploss + trailing for longs
    # ═════════════════════════════════════════════════════════════════════════
    #
    # CRITICAL: Freqtrade's backtesting entry-candle exit price formula differs
    # between built-in trailing and custom_stoploss:
    #   Built-in: stop_rate = OPEN * (1 + offset - trail/leverage)
    #     → At 15x, offset=0.05 dominates → price far ABOVE HIGH → unfillable → trade survives
    #   Custom:   stop_rate = OPEN * (1 - stop_loss_pct/leverage)
    #     → stop_loss_pct is tiny (0.02) → price just below OPEN → fillable → trade killed
    #
    # Fix: skip trailing on the entry candle (< 1 timeframe). This matches the
    # built-in trailing's effective behavior where the entry-candle formula prevents
    # exit on that candle.
    #
    def custom_stoploss(
        self,
        pair,
        trade,
        current_time,
        current_rate,
        current_profit,
        after_fill,
        **kwargs,
    ):
        if trade.is_short:
            # Short: fixed stoploss at -35% from open (NinjaForgeShort's stoploss)
            return (
                stoploss_from_open(
                    -0.35, current_profit, is_short=True, leverage=trade.leverage
                )
                or 1
            )

        # Long: trailing stoploss matching NinjaFutures5m's built-in trailing
        # Base: -20%, trailing activates at 5% profit, trails 2% behind peak

        # On the entry candle, only apply base stoploss — no trailing.
        # This prevents the backtesting entry-candle exit formula from producing
        # a fillable exit price that kills the trade immediately (see note above).
        trade_age_s = (current_time - trade.open_date_utc).total_seconds()
        if trade_age_s < 300:  # < 1 candle (5m timeframe)
            return (
                stoploss_from_open(
                    self.LONG_STOPLOSS,
                    current_profit,
                    is_short=False,
                    leverage=trade.leverage,
                )
                or 1
            )

        max_rate = trade.max_rate if trade.max_rate is not None else current_rate
        effective_max_rate = max(max_rate, current_rate)
        if trade.open_rate > 0:
            max_profit = (effective_max_rate / trade.open_rate - 1) * trade.leverage
        else:
            max_profit = current_profit

        # Once offset reached (5% leveraged profit), start trailing
        if max_profit >= self.LONG_TRAILING_OFFSET:
            desired_stop = max_profit - self.LONG_TRAILING_STOP
            sl = stoploss_from_open(
                desired_stop, current_profit, is_short=False, leverage=trade.leverage
            )
            # 'or 1' is the standard Freqtrade idiom: when stoploss_from_open
            # returns 0.0 (stop would be above current price), 0.0 is falsy and
            # would skip adjust_stop_loss. Returning 1 (max distance) instead
            # keeps the previous stop in place via the ratchet-only rule.
            return sl or 1

        # Below offset: fixed at -20%
        return (
            stoploss_from_open(
                self.LONG_STOPLOSS,
                current_profit,
                is_short=False,
                leverage=trade.leverage,
            )
            or 1
        )

    # ═════════════════════════════════════════════════════════════════════════
    # PER-SIDE STOPLOSS GUARD — replaces built-in StoplossGuard
    # ═════════════════════════════════════════════════════════════════════════
    def _is_sl_guard_active(
        self,
        current_time,
        side: str,
        lookback_hours: float,
        trade_limit: int,
        pause_hours: float,
        per_pair: bool = False,
        pair: str = None,
    ) -> bool:
        """Per-side StoplossGuard evaluator.
        Returns True when same-side stoploss clusters trigger an active pause window.
        """
        search_start = current_time - timedelta(hours=lookback_hours + pause_hours)
        all_recent = Trade.get_trades_proxy(is_open=False, close_date=search_start)

        # Filter by side + stoploss exit reason
        sl_trades = [
            t
            for t in all_recent
            if (t.is_short == (side == "short"))
            and t.exit_reason in ("stop_loss", "stoploss_on_exchange")
        ]

        if per_pair and pair:
            sl_trades = [t for t in sl_trades if t.pair == pair]

        if len(sl_trades) < trade_limit:
            return False

        # Sort chronologically and check sliding windows
        sl_trades.sort(key=lambda t: t.close_date_utc)

        for i in range(len(sl_trades) - trade_limit + 1):
            group = sl_trades[i : i + trade_limit]
            window_span_h = (
                group[-1].close_date_utc - group[0].close_date_utc
            ).total_seconds() / 3600

            if window_span_h <= lookback_hours:
                # Guard triggered at last SL in this group
                pause_until = group[-1].close_date_utc + timedelta(hours=pause_hours)
                if current_time < pause_until:
                    return True

        return False

    # ═════════════════════════════════════════════════════════════════════════
    # CONFIRM TRADE ENTRY — per-side protections + short's post-SL gate
    # ═════════════════════════════════════════════════════════════════════════
    def confirm_trade_entry(
        self,
        pair,
        order_type,
        amount,
        rate,
        time_in_force,
        current_time,
        entry_tag,
        side,
        **kwargs,
    ) -> bool:
        """Apply manual per-side StoplossGuard logic before entry, plus short post-SL bear-regime gating."""
        # ── LONG PROTECTIONS (from NinjaFutures5m V14.0) ──
        if side == "long":
            # StoplossGuard: 1 long SL in 4h → block long 6h (all pairs)
            if self._is_sl_guard_active(
                current_time,
                "long",
                lookback_hours=self.LONG_SL_GUARD_LOOKBACK_H,
                trade_limit=self.LONG_SL_GUARD_TRADE_LIMIT,
                pause_hours=self.LONG_SL_GUARD_PAUSE_H,
            ):
                return False
            return True

        # ── SHORT PROTECTIONS (from NinjaForgeShort v16.0) ──

        # Tier 1: Per-pair StoplossGuard — 2 SLs in 2h → block pair 2h
        if self._is_sl_guard_active(
            current_time,
            "short",
            lookback_hours=self.SHORT_SL_GUARD_PAIR_LOOKBACK_H,
            trade_limit=self.SHORT_SL_GUARD_PAIR_TRADE_LIMIT,
            pause_hours=self.SHORT_SL_GUARD_PAIR_PAUSE_H,
            per_pair=True,
            pair=pair,
        ):
            return False

        # Tier 2: Cascade StoplossGuard — 2 SLs in 10h → block all shorts 12h
        if self._is_sl_guard_active(
            current_time,
            "short",
            lookback_hours=self.SHORT_SL_GUARD_CASCADE_LOOKBACK_H,
            trade_limit=self.SHORT_SL_GUARD_CASCADE_TRADE_LIMIT,
            pause_hours=self.SHORT_SL_GUARD_CASCADE_PAUSE_H,
        ):
            return False

        # Tier 3: Portfolio StoplossGuard — 3 SLs in 24h → block all shorts 18h
        if self._is_sl_guard_active(
            current_time,
            "short",
            lookback_hours=self.SHORT_SL_GUARD_PORTFOLIO_LOOKBACK_H,
            trade_limit=self.SHORT_SL_GUARD_PORTFOLIO_TRADE_LIMIT,
            pause_hours=self.SHORT_SL_GUARD_PORTFOLIO_PAUSE_H,
        ):
            return False

        # Post-SL gate: after recent large SL loss, require full bear regime
        sl_lookback = current_time - timedelta(hours=self.POST_SL_GATE_HOURS)
        all_recent = Trade.get_trades_proxy(is_open=False, close_date=sl_lookback)
        has_recent_sl = any(
            t.is_short and t.close_profit is not None and t.close_profit <= -0.30
            for t in all_recent
        )
        if has_recent_sl:
            dataframe, _ = self.dp.get_analyzed_dataframe(
                pair=pair, timeframe=self.timeframe
            )
            if len(dataframe) >= 1:
                candle = dataframe.iloc[-1]
                if candle.get("bear_regime_1h", 0) != 1:
                    return False

        return True

    # ═════════════════════════════════════════════════════════════════════════
    # CUSTOM EXIT — routes to per-side exit logic
    # ═════════════════════════════════════════════════════════════════════════
    def custom_exit(
        self, pair, trade, current_time, current_rate, current_profit, **kwargs
    ):
        """Route exit evaluation to long or short custom exit logic by trade direction."""
        if trade.is_short:
            return self._short_custom_exit(
                pair, trade, current_time, current_rate, current_profit, **kwargs
            )
        else:
            return self._long_custom_exit(
                pair, trade, current_time, current_rate, current_profit, **kwargs
            )

    # ─────────────────────────────────────────────────────────────────────
    # LONG EXIT — from NinjaFutures5m (fastk_profit_sell, cci_recovery_sell)
    # ─────────────────────────────────────────────────────────────────────
    def _long_custom_exit(
        self, pair, trade, current_time, current_rate, current_profit, **kwargs
    ):
        """Long exits: take overbought profit via fastk or exit post-drawdown recovery via CCI."""
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        if len(dataframe) == 0:
            return None
        current_candle = dataframe.iloc[-1].squeeze()

        # fastk profit sell — in profit + StochFast K > 90 (overbought)
        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx:
                return "fastk_profit_sell"

        # CCI recovery sell — deep dip recovered + CCI confirms
        min_profit = trade.calc_profit_ratio(trade.min_rate)
        if min_profit <= -0.10:
            if current_profit > self.sell_cci_profit:
                if current_candle["cci"] > self.sell_cci_threshold:
                    return "cci_recovery_sell"

        return None

    # ─────────────────────────────────────────────────────────────────────
    # SHORT EXIT — from NinjaForgeShort (deferral wrapper + R-multiple stack)
    # ─────────────────────────────────────────────────────────────────────
    def _short_custom_exit(
        self, pair, trade, current_time, current_rate, current_profit, **kwargs
    ):
        """Run short R-stack evaluation first, then optionally defer eligible winner exits."""
        exit_result = self._evaluate_short_exit(
            pair, trade, current_time, current_rate, current_profit, **kwargs
        )

        # Winner exit deferral (bear_bounce only, not correction_fade)
        tag = trade.enter_tag or ""
        if (
            exit_result is not None
            and "slope_bounce" not in tag
            and "correction_fade" not in tag
        ):
            if trade.open_rate > 0:
                price_move = current_profit / trade.leverage
                r_multiple = price_move / self.TREND_STOP_PCT

                dataframe, _ = self.dp.get_analyzed_dataframe(
                    pair=pair, timeframe=self.timeframe
                )
                if len(dataframe) >= 1:
                    candle = dataframe.iloc[-1]
                    bear_1h = candle.get("bear_regime_1h", 0)
                    adx_5m = candle.get("adx", 0)

                    is_deferrable = (
                        bear_1h == 1
                        and exit_result in self.DEFER_PROFIT_EXITS
                        and r_multiple >= self.DEFER_MIN_R
                        and adx_5m >= self.DEFER_ADX_MIN
                    )

                    defer_ts = trade.get_custom_data("defer_start_ts", default=None)

                    if is_deferrable:
                        if defer_ts is None:
                            trade.set_custom_data(
                                "defer_start_ts", current_time.timestamp()
                            )
                            return None
                        minutes_elapsed = (current_time.timestamp() - defer_ts) / 60
                        if minutes_elapsed < self.DEFER_MAX_MINUTES:
                            if r_multiple < self.DEFER_CANCEL_R:
                                trade.set_custom_data("defer_start_ts", None)
                                return exit_result
                            return None
                        trade.set_custom_data("defer_start_ts", None)
                        return exit_result
                    elif defer_ts is not None:
                        trade.set_custom_data("defer_start_ts", None)
                        return exit_result

        return exit_result

    # ─────────────────────────────────────────────────────────────────────
    # SHORT EXIT EVALUATION — R-multiple priority stack (verbatim from NinjaForgeShort)
    # ─────────────────────────────────────────────────────────────────────
    def _evaluate_short_exit(
        self, pair, trade, current_time, current_rate, current_profit, **kwargs
    ):
        """Evaluate short exits in priority order using regime invalidation, R-thresholds, and time-based failsafes."""
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        if len(dataframe) < 2:
            return None
        candle = dataframe.iloc[-1]
        prev_candle = dataframe.iloc[-2]
        trade_dur = current_time - trade.open_date_utc
        trade_hours = trade_dur.total_seconds() / 3600
        if trade.open_rate <= 0:
            return None
        stop_pct = self.TREND_STOP_PCT
        price_move_pct = current_profit / trade.leverage
        r_multiple = price_move_pct / stop_pct

        bull_1h = candle.get("bull_regime_1h", 0)
        bear_1h = candle.get("bear_regime_1h", 0)
        correction_1h = candle.get("correction_regime_1h", 0)
        bull_accel_1h = candle.get("bull_accel_regime_1h", 0)

        tag = trade.enter_tag or ""
        is_corr_fade = "correction_fade" in tag

        bb_exit_r = self.TREND_BB_EXIT_R
        exhaustion_r = self.TREND_EXHAUSTION_R
        tp_r = self.CORR_TP_SHORT_R if is_corr_fade else self.TREND_TP_SHORT_R
        max_hold = self.CORR_MAX_HOLD if is_corr_fade else self.TREND_SHORT_MAX_HOLD

        # Extended hold in deep bear (bear_bounce only)
        if bear_1h == 1 and not is_corr_fade:
            bear_streak = candle.get("bear_streak_1h", 0)
            if bear_streak >= 6:
                max_hold = 8

        # Invalidation exit — regime shift to bull
        if trade_hours >= 1.0:
            regime_dead = bear_1h == 0 and (bull_1h == 1 or bull_accel_1h == 1)
            if regime_dead and current_profit > self.INVALIDATION_PROFIT_GATE:
                if (
                    candle["rsi_14"] <= 50
                    and current_profit > self.INVALIDATION_RSI_GATE_PROFIT
                ):
                    pass
                else:
                    return "invalidation_exit"

        # Correction fade: ST invalidation
        if is_corr_fade and trade_hours >= 0.5:
            st_dir = candle.get("st_direction_1h", 0)
            if st_dir == 1 and current_profit > -0.15:
                return "corr_st_invalidation_exit"

        # Trend fade exit
        fade_gate = -0.15 if is_corr_fade else self.TREND_FADE_PROFIT_GATE
        if current_profit < fade_gate:
            if trade_hours >= self.TREND_FADE_MIN_HOURS and bear_1h == 0:
                if (
                    candle["adx"] < self.TREND_FADE_ADX_MAX
                    and candle["adx"] < prev_candle["adx"]
                    and candle["macd_hist"] > 0
                ):
                    return "trend_fade_exit"

        # BB lower exit
        if r_multiple >= bb_exit_r and candle["close"] < candle["bb_lower"]:
            return "bb_lower_exit"

        # Exhaustion exits
        if r_multiple >= exhaustion_r:
            if prev_candle["fastk"] < 20 and candle["fastk"] > 20:
                return "stoch_exhaustion_exit"
            if prev_candle["willr"] < -80 and candle["willr"] > -80:
                return "willr_exhaustion_exit"
            if prev_candle["macd_hist"] < 0 and candle["macd_hist"] > 0:
                return "macd_exhaustion_exit"

        # Take profit
        if r_multiple >= tp_r:
            return f"take_profit_{tp_r}R"

        # Stagnation exit
        if is_corr_fade:
            stagnation_hours = self.CORR_STAGNATION_HOURS
        else:
            stagnation_hours = 5 if bear_1h == 1 else 4
        if trade_hours >= stagnation_hours and -0.1 <= r_multiple <= 0.5:
            recent = dataframe.tail(12)
            if len(recent) >= 12:
                price_range = recent["high"].max() - recent["low"].min()
                avg_atr = recent["atr"].mean()
                if avg_atr > 0 and price_range < avg_atr:
                    return "stagnation_exit"
            if candle["chop"] > 60 and candle["adx"] < 18:
                return "chop_stagnation_exit"

        # Momentum exit
        if trade_hours >= 3.0 and current_profit < self.MOM_EXIT_SHORT:
            if (
                candle["rsi_14"] > 55
                and candle["fastk"] > 70
                and candle["macd_hist"] > 0
                and candle["adx"] < prev_candle["adx"]
            ):
                return "momentum_exit"

        # Time exit — correction regime cut
        corr_time_cut = 4.0 if is_corr_fade else 6.0
        if trade_hours >= corr_time_cut and correction_1h == 1 and current_profit < 0:
            return "time_exit_cut"

        # Time exit — max hold
        if trade_hours >= max_hold:
            if current_profit > 0:
                return "time_exit_profit"
            return "time_exit_cut"

        return None
