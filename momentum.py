from pykrx import stock
import pandas as pd
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")

def calculate_signal(df):
    M0 = df['close'].rolling(window=5).mean()
    M1 = df['close'].rolling(window=10).mean()
    M2 = df['close'].rolling(window=20).mean()
    M3 = df['close'].rolling(window=60).mean()
    M4 = df['close'].rolling(window=120).mean()

    m0ang = M0.diff()
    m1ang = M1.diff()
    m2ang = M2.diff()
    m3ang = M3.diff()
    m4ang = M4.diff()

    m0s = (M0 >= M0.shift(1)).astype(int).replace(0, -1)
    m1s = (M1 >= M1.shift(1)).astype(int).replace(0, -1)
    m2s = (M2 >= M2.shift(1)).astype(int).replace(0, -1)
    m3s = ((M3 >= M3.shift(1)) | (m3ang > -2)).astype(int).replace(False, -1)
    m4s = ((M4 >= M4.shift(1)) | (m4ang > -1)).astype(int).replace(False, -1)

    m3sm = ((M3 <= M3.shift(1)) | (m3ang < 2)).astype(int).replace(False, -1) * -1
    m4sm = ((M4 <= M4.shift(1)) | (m4ang < 1)).astype(int).replace(False, -1) * -1

    close = df['close']
    s1 = (close >= M0).astype(int).replace(False, -1)
    s2 = (close >= M1).astype(int).replace(False, -1)
    s3 = (close >= M2).astype(int).replace(False, -1)
    s4 = (close >= M3).astype(int).replace(False, -1)
    s5 = (close >= M4).astype(int).replace(False, -1)

    jung = pd.Series(0, index=df.index)
    cond2 = (close >= M0) & (M0 >= M1) & (M1 >= M2) & (M2 >= M3) & (M3 >= M4)
    cond1 = (close >= M0) & (M0 >= M1) & (M1 >= M2) & (M2 >= M3)
    jung.loc[cond2] = 2
    jung.loc[~cond2 & cond1] = 1

    HLd99 = pd.Series(0, index=df.index)
    cond_HLd99_2 = (m1s == 1) & (m2s == 1) & (m3s == 1)
    cond_HLd99_1 = (m1s == 1) & (m2s == 1)
    cond_HLd99_m2 = (m0s == -1) & (m1s == -1) & (m2s == -1) & (m3sm == -1)
    cond_HLd99_m1 = (m1s == -1) & (m2s == -1)

    HLd99.loc[cond_HLd99_2] = 2
    HLd99.loc[~cond_HLd99_2 & cond_HLd99_1] = 1
    HLd99.loc[cond_HLd99_m2] = -2
    HLd99.loc[cond_HLd99_m1 & ~cond_HLd99_m2] = -1

    HLv99 = HLd99.replace(to_replace=0, method='ffill').fillna(0)

    new_high_flag = 0
    if len(df) >= 126:
        max_recent_3 = df['close'].iloc[-3:].max()
        max_past_6m = df['close'].iloc[-126:].max()
        if max_recent_3 >= max_past_6m:
            new_high_flag = 1
    else:
        max_recent_3 = df['close'].iloc[-3:].max()
        max_past = df['close'].max()
        if max_recent_3 >= max_past:
            new_high_flag = 1

    sco99 = (
        m0s + m1s + m2s + m3s + m4s +
        s1 + s2 + s3 + s4 + s5 +
        jung + HLd99 +
        new_high_flag
    )
    sco = sco99.rolling(window=4).mean()
    df['sco'] = sco
    return df

def normalize_0_1(series):
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)

def main():
    tickers = [
        '091160', '091180', '305720', '117460', '244580', '091170',
        '102970', '117680', '117700', '139230', '228810', '228790',
        '069500', '229200', '487230'
    ]
    industry_kor = [
        '반도체', '자동차', '이차전', '에너지', '바이오', '은행주',
        '증권주', '철강주', '건설주', '조선주', '엔터주', '화장품',
        '코스피', '코스닥', '전력인'
    ]
    industry_map = dict(zip(tickers, industry_kor))

    end = datetime.today()
    start = end - timedelta(days=180)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    results = []
    for code in tickers:
        df = stock.get_market_ohlcv_by_date(start_str, end_str, code)
        if df.empty or len(df) < 60:
            continue

        name = stock.get_market_ticker_name(code)
        industry = industry_map.get(code, 'Unknown')
        old = df['종가'].iloc[0]
        new = df['종가'].iloc[-1]
        rtn = (new / old - 1) * 100

        df_signal = df.copy()
        df_signal.rename(columns={'종가': 'close'}, inplace=True)
        df_signal = calculate_signal(df_signal)
        if len(df_signal) > 1:
            today_close = df_signal['close'].iloc[-1]
            prev_close = df_signal['close'].iloc[-2]
            today_chg = (today_close / prev_close - 1) * 100 if prev_close != 0 else 0
        else:
            today_chg = 0
        
        latest_sco = df_signal['sco'].dropna().iloc[-1] if not df_signal['sco'].dropna().empty else None

        if (latest_sco is None) or (rtn is None):
            continue

        results.append({
            '티커': code,
            '종목명': name,
            '산업': industry,
            '수익률(%)': rtn,
            'Signal_sco': latest_sco,
            '당일등락률(%)': today_chg 
        })

    df_result = pd.DataFrame(results)
    df_result['Norm_sco'] = normalize_0_1(df_result['Signal_sco'])
    df_result['Norm_rtn'] = normalize_0_1(df_result['수익률(%)'])
    w_sco = 80
    w_rtn = 20
    df_result['Final_score'] = w_sco * df_result['Norm_sco']+ w_rtn * df_result['Norm_rtn']
    df_result = df_result.sort_values(by='Final_score', ascending=False).reset_index(drop=True)

    total_investment = 30_000_000

    # 상위 5개 선택
    top5 = df_result.head(5).copy()
    top5_score_sum = top5['Final_score'].sum()

    # 초기 투자금액 컬럼 0으로 세팅
    df_result['투자금액'] = 0

    # 상위 5개에만 투자금액 배분
    df_result.loc[top5.index, '투자금액'] = ((top5['Final_score'] / top5_score_sum) * total_investment).round(0)

    # 출력 너비 설정
    width_rank = 3
    width_ticker = 6
    width_industry = 8
    width_sco = 10
    width_rtn = 10
    width_final = 10
    width_amt = 10

    print("\n상위 5개 추천 (최종 점수 기준, sco 50% + 3M 수익률 50%, 전액 분배):")
    for i, row in df_result.head(5).iterrows():
        rank_str = f"{i+1}.".ljust(width_rank)
        ticker_str = f"{row['티커']}".ljust(width_ticker)
        industry_str = f"{row['산업']:<10s}".ljust(width_industry)
        today_chg_str = f"{row['당일등락률(%)']:+.1f}%"
        sco_str = f"sco: {row['Signal_sco']:.1f}".rjust(width_sco)
        rtn_str = f"3M: {row['수익률(%)']:.1f}%".rjust(width_rtn)
        final_str = f"최종점수: {row['Final_score']:.0f}".rjust(width_final)
        amt_str = f"투자금액: {int(row['투자금액']):,}원".rjust(width_amt)
        print(f"{rank_str} {ticker_str} {industry_str}{today_chg_str} {sco_str}, {rtn_str}, {final_str}, {amt_str}")

    print("\n전체 종목 (최종 점수 내림차순, 상위 5개 투자금액 반영):")
    for i, row in df_result.iloc[5:].iterrows():
        rank_str = f"{i+1}.".ljust(width_rank)
        ticker_str = f"{row['티커']}".ljust(width_ticker)
        industry_str = f"{row['산업']}".ljust(width_industry)
        today_chg_str = f"{row['당일등락률(%)']:+.1f}%"
        sco_str = f"sco: {row['Signal_sco']:.1f}".rjust(width_sco)
        rtn_str = f"3M: {row['수익률(%)']:.1f}%".rjust(width_rtn)
        final_str = f"점수: {row['Final_score']:.0f}".rjust(width_final)
        amt_str = f"투자금액: {int(row['투자금액']):,}원".rjust(width_amt)
        print(f"{rank_str} {ticker_str}{industry_str}{today_chg_str} {sco_str}, {rtn_str}, {final_str}, {amt_str}")

if __name__ == "__main__":
    main()

input("\n프로그램이 종료되었습니다. 결과를 확인한 후 Enter 키를 눌러 창을 닫으세요.")
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(df_result.to_string())  # DataFrame을 문자열 형태로 변환해 쓰기
input("\nresult.txt를 확인하세요. Enter를 누르면 종료합니다.")

