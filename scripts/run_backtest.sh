#!/usr/bin/env bash
# Run backtesting for NinjaCombined strategy
# Usage: ./scripts/run_backtest.sh [compound|flat] [period]
#   compound (default): $1000 start, 75% wallet per trade
#   flat:               $1000 start, $100 fixed per trade
#   period: 1, 2, 3, oos, or all (default: all)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODE="${1:-compound}"
PERIOD="${2:-all}"

if [[ "$MODE" == "compound" ]]; then
    CONFIG="config_compound.json"
    LABEL="Compound ($1000, 75%)"
elif [[ "$MODE" == "flat" ]]; then
    CONFIG="config_flat.json"
    LABEL="Flat ($1000, $100/trade)"
else
    echo "Usage: $0 [compound|flat] [1|2|3|oos|all]"
    exit 1
fi

run_backtest() {
    local timerange="$1"
    local period_label="$2"

    echo "═══════════════════════════════════════════════════════════════"
    echo "  ${LABEL} — ${period_label}"
    echo "  Timerange: ${timerange}"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    docker run --rm \
        -v "${PROJECT_DIR}/user_data:/freqtrade/user_data" \
        auto-bot-merge:latest backtesting \
        --config "/freqtrade/user_data/${CONFIG}" \
        --strategy NinjaCombined \
        --timerange "${timerange}" \
        --cache none \
        --enable-protections 2>&1 | tail -80

    echo ""
}

case "$PERIOD" in
    1)   run_backtest 20200710-20220510 "P1 Bull→Crash (2020-07-10 ~ 2022-05-10)" ;;
    2)   run_backtest 20220510-20240310 "P2 Bear→Recovery (2022-05-10 ~ 2024-03-10)" ;;
    3)   run_backtest 20240310-20251231 "P3 ETF Rally→Correction (2024-03-10 ~ 2025-12-31)" ;;
    oos) run_backtest 20260101-20260303 "OOS Holdout (2026-01-01 ~ 2026-03-03)" ;;
    all)
        run_backtest 20200710-20220510 "P1 Bull→Crash (2020-07-10 ~ 2022-05-10)"
        run_backtest 20220510-20240310 "P2 Bear→Recovery (2022-05-10 ~ 2024-03-10)"
        run_backtest 20240310-20251231 "P3 ETF Rally→Correction (2024-03-10 ~ 2025-12-31)"
        run_backtest 20260101-20260303 "OOS Holdout (2026-01-01 ~ 2026-03-03)"
        ;;
    *)   echo "Unknown period: $PERIOD (use 1, 2, 3, oos, or all)"; exit 1 ;;
esac

echo "=== Backtest Complete ==="
