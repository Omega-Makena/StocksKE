import pandas as pd
from sqlalchemy.orm import Session
from app.models.price import Price
from app.models.sentiment import SentimentScore
from datetime import datetime, timedelta

def build_features_for_ticker(db: Session, ticker_id: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Builds a time-indexed feature dataframe for a ticker.
    Features: Returns (1d, 5d, 20d), SMA, RSI, Sentiment (mean, trend).
    """
    prices = db.query(Price).filter(
        Price.ticker_id == ticker_id,
        Price.time >= start_date - timedelta(days=60), # Extra days for rolling logic
        Price.time <= end_date
    ).order_by(Price.time.asc()).all()
    
    if not prices:
        return pd.DataFrame()
        
    df = pd.DataFrame([{
        "time": p.time,
        "close": p.close,
        "volume": p.volume
    } for p in prices])
    df.set_index("time", inplace=True)
    
    # Technicals
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)
    df['vol_change'] = df['volume'].pct_change(1)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Simplified RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Target (next day direction)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Sentiments
    sentiments = db.query(SentimentScore).filter(
        SentimentScore.ticker_id == ticker_id,
        SentimentScore.created_at >= start_date - timedelta(days=10),
        SentimentScore.created_at <= end_date
    ).all()
    
    s_df = pd.DataFrame([{
        "time": s.created_at.date(), # Approximate grouping by day
        "score": s.sentiment_score
    } for s in sentiments])
    
    if not s_df.empty:
        s_df['time'] = pd.to_datetime(s_df['time'])
        s_daily = s_df.groupby('time')['score'].mean().reset_index()
        s_daily.set_index('time', inplace=True)
        # Merge back to price df (ensure datetime types align)
        df = df.join(s_daily, how='left')
        df['score'] = df['score'].fillna(0) # Neutral score where no news
        df['score_trend_3d'] = df['score'].rolling(3).mean()
    else:
        df['score'] = 0.0
        df['score_trend_3d'] = 0.0
        
    df.dropna(inplace=True)
    df = df[df.index >= start_date] # Trim warmup rows
    
    return df
