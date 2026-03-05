# NinjaCombined — Freqtrade 양방향 자동매매 봇

NinjaFutures5m V14.0 (롱) + NinjaForgeShort (숏) 전략을 하나로 합친 Binance USDT-M Futures 자동매매 봇.

- **1 포지션만** 동시 보유 (`max_open_trades: 1`)
- **진입 시 잔고의 75%** 사용 (`tradable_balance_ratio: 0.75`)
- **레버리지 15x**, Isolated Margin
- **5분봉** 기반

---

## 사전 준비

### 1. Binance 계정 설정

1. **Futures 계정 활성화** — Binance 앱/웹에서 선물 계정 개설
2. **포지션 모드**: **One-way Mode** (단방향 모드)로 설정
   - Binance Futures → 설정(⚙️) → Position Mode → One-way Mode
   - ⚠️ Hedge Mode 사용 시 봇이 정상 작동하지 않음
3. **자산 모드**: **Single-Asset Mode** (단일 자산 모드)
   - Binance Futures → 설정(⚙️) → Asset Mode → Single-Asset Mode
4. **API Key 생성**
   - Binance → API Management → Create API
   - **Enable Futures** 체크 필수
   - IP 제한(whitelist) 권장 — 봇 서버 IP만 허용
   - ⚠️ 출금(Withdraw) 권한은 **절대 주지 마세요**

### 2. Telegram Bot (선택, 강력 권장)

1. [@BotFather](https://t.me/BotFather)에서 봇 생성 → **Token** 획득
2. 생성된 봇에게 아무 메시지 전송
3. `https://api.telegram.org/bot<TOKEN>/getUpdates` 접속 → **chat_id** 획득

### 3. Docker & Docker Compose 설치

```bash
# macOS
brew install --cask docker

# Linux (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sh
sudo apt install docker-compose-plugin
```

---

## 설치 및 설정

### 1. 프로젝트 클론

```bash
git clone <repo-url> auto-bot-merge
cd auto-bot-merge
```

### 2. 시크릿 설정

```bash
cp user_data/config_secrets.json.example user_data/config_secrets.json
```

`user_data/config_secrets.json`을 열고 실제 값을 입력:

```json
{
    "exchange": {
        "key": "실제_바이낸스_API_KEY",
        "secret": "실제_바이낸스_API_SECRET"
    },
    "telegram": {
        "enabled": true,
        "token": "실제_텔레그램_봇_TOKEN",
        "chat_id": "실제_채팅_ID"
    }
}
```

> ⚠️ `config_secrets.json`은 `.gitignore`에 포함되어 있어 git에 올라가지 않습니다.

### 3. Docker 이미지 빌드

```bash
docker compose build
```

---

## 실매매 시작

### 실매매 실행

```bash
docker compose up -d live
```

- `restart: unless-stopped` — 서버 재부팅 시 자동 재시작
- 로그 자동 로테이션 (50MB × 5개)

### 실매매 중지

```bash
docker compose down
```

> 포지션이 열려있을 때 중지하면 포지션이 그대로 남습니다.
> 포지션을 먼저 정리하고 싶다면 텔레그램에서 `/forceexit all` 명령 후 중지하세요.

---

## 모의매매 (Dry Run)

실제 돈 없이 동일 로직 테스트:

```bash
# 시작
docker compose --profile dryrun up -d dryrun

# 중지
docker compose --profile dryrun down
```

---

## 모니터링

### 로그 확인

```bash
# 실시간 로그
docker logs -f ninja-combined-live

# 최근 100줄
docker logs --tail 100 ninja-combined-live
```

### Telegram 명령어 (주요)

| 명령어 | 설명 |
|--------|------|
| `/status` | 현재 열린 포지션 상태 |
| `/profit` | 수익 요약 |
| `/balance` | 계정 잔고 |
| `/forceexit <trade_id>` | 특정 포지션 강제 청산 |
| `/forceexit all` | 모든 포지션 강제 청산 |
| `/stopentry` | 새 진입 중지 (기존 포지션은 유지) |
| `/reload_config` | 설정 리로드 |
| `/help` | 전체 명령어 보기 |

### 컨테이너 상태 확인

```bash
docker compose ps
```

---

## 백테스트

```bash
# 데이터 다운로드 (처음 1회)
docker compose run --rm backtest download-data \
  --config /freqtrade/user_data/config_compound.json \
  --timerange 20200101- \
  --timeframe 5m 1h

# 백테스트 실행
docker compose run --rm backtest

# 커스텀 기간
docker compose run --rm backtest backtesting \
  --config /freqtrade/user_data/config_compound.json \
  --strategy NinjaCombined \
  --timerange 20240101-20250101
```

---

## 프로젝트 구조

```
auto-bot-merge/
├── docker-compose.yml              # 서비스 정의 (live, dryrun, backtest)
├── docker/
│   └── Dockerfile.custom           # freqtrade:stable + pandas-ta
├── requirements.txt                # pandas-ta==0.4.71b0
├── scripts/
│   └── run_backtest.sh             # 백테스트 헬퍼 스크립트
├── user_data/
│   ├── strategies/
│   │   └── NinjaCombined.py        # 메인 전략 (롱+숏)
│   ├── config_live.json            # 실매매 설정
│   ├── config_compound.json        # 백테스트 설정 (복리)
│   ├── config_flat.json            # 백테스트 설정 (단리)
│   ├── config_secrets.json         # API 키 (gitignored)
│   └── config_secrets.json.example # API 키 템플릿
└── .gitignore
```

---

## 주요 설정값 (config_live.json)

| 항목 | 값 | 설명 |
|------|-----|------|
| `dry_run` | `false` | 실매매 모드 |
| `max_open_trades` | `1` | 동시 1포지션 |
| `tradable_balance_ratio` | `0.75` | 잔고의 75% 사용 |
| `stake_amount` | `"unlimited"` | 비율 기반 진입 |
| `trading_mode` | `"futures"` | USDT-M 선물 |
| `margin_mode` | `"isolated"` | 격리 마진 |
| `liquidation_buffer` | `0.05` | 청산가 대비 5% 버퍼 |
| `stoploss_on_exchange` | `true` | 거래소 스탑로스 |
| `order_types` | `"market"` (전체) | 시장가 주문 |

---

## 전략 요약

### 롱 (NinjaFutures5m V14.0)
- 4개 진입 시그널: `clucHA`, `vwap`, `SVWAP`, `cofi`
- `custom_stoploss`: 기본 -0.20, 트레일링 (offset 0.04 / stop 0.02)
- RSI, Bollinger Band, VWAP, EMA 기반

### 숏 (NinjaForgeShort)
- 2개 진입 시그널: `short_rsi_bb`, `short_stoch_rsi`
- 기본 stoploss -0.35 (custom_stoploss에서 롱만 오버라이드)
- ROI 기반 청산 (0: 0.15, 10: 0.10, 20: 0.07, 60: 0.05, 120: 0.03)

---

## 주의사항

- **실제 자금이 투입됩니다.** 충분한 dry run 테스트 후 실매매를 권장합니다.
- 과거 백테스트 수익이 미래 수익을 보장하지 않습니다.
- 레버리지 15x는 높은 수준입니다. 감당 가능한 금액만 투입하세요.
- 서버/네트워크 장애 시 `stoploss_on_exchange`가 안전장치 역할을 합니다.
- API Key에 출금 권한을 부여하지 마세요.
