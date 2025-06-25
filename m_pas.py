import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import time
import requests
from dotenv import load_dotenv
import os

# تحميل مفاتيح API من ملف .env
load_dotenv()

# إعدادات الصفحة
st.set_page_config(
    page_title="نظام تحليل الأسهم المتقدم",
    layout="wide",
    initial_sidebar_state="expanded"
)

# عنوان التطبيق
st.title('📈 نظام تحليل الأسهم الأمريكي المتقدم')
st.markdown("""
هذا التطبيق يقوم بما يلي:
- عرض الأخبار المالية العاجلة
- تتبع الأسهم الأكثر ارتفاعاً
- التحليل الفني المتقدم
- التنبؤ بحركة الأسهم باستخدام الذكاء الاصطناعي
""")

# API Keys (يتم تحميلها من ملف .env)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "f55929edb5ee471791a1e622332ff6d8")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY", "16be092ddfdcb6e34f1de36875a3072e2c724afb")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "X5QLR930PG6ONM5H")

# ---------------------------------------------------
# قسم الأخبار العاجلة
# ---------------------------------------------------
@st.cache_data(ttl=3600)
def get_financial_news():
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        news = newsapi.get_top_headlines(
            category='business',
            language='en',
            country='us'
        )
        return news.get('articles', [])
    except Exception as e:
        st.error(f"حدث خطأ في جلب الأخبار: {str(e)}")
        return []

# ---------------------------------------------------
# قسم الأسهم الأكثر ارتفاعاً (بدائل متعددة)
# ---------------------------------------------------
@st.cache_data(ttl=3600)
def get_top_gainers():
    # المحاولة باستخدام Tiingo أولاً
    tiingo_data = get_top_gainers_tiingo()
    if not tiingo_data.empty:
        return tiingo_data
    
    # إذا فشل، جرب Yahoo Finance
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        url = "https://finance.yahoo.com/gainers"
        tables = pd.read_html(url, header=0, attrs={'headers': headers})
        return tables[0].head(10) if tables else pd.DataFrame()
    except Exception as e:
        st.error(f"حدث خطأ في جلب الأسهم الصاعدة: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_top_gainers_tiingo():
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {TIINGO_API_KEY}'
        }
        response = requests.get(
            "https://api.tiingo.com/tiingo/daily/top",
            headers=headers,
            params={'columns': 'ticker,priceChange,priceChangePercent,volume'}
        )
        data = response.json()
        return pd.DataFrame(data).head(10)
    except Exception as e:
        st.warning(f"لم يتمكن من جلب البيانات من Tiingo: {str(e)}")
        return pd.DataFrame()

# ---------------------------------------------------
# وظائف التحليل الفني المحسنة
# ---------------------------------------------------
def calculate_technical_indicators(data):
    try:
        if data.empty or 'Close' not in data.columns:
            return data
            
        # حساب مؤشر RSI مع معالجة القيم الفارغة
        data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi().fillna(50)
        
        # حساب Bollinger Bands
        bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
        data['BB_high'] = bb_indicator.bollinger_hband().fillna(data['Close'])
        data['BB_low'] = bb_indicator.bollinger_lband().fillna(data['Close'])
        data['BB_mid'] = (data['BB_high'] + data['BB_low']) / 2
        
        # حساب MACD
        macd_indicator = MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd_indicator.macd().fillna(0)
        data['MACD_signal'] = macd_indicator.macd_signal().fillna(0)
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']
        
        return data
    except Exception as e:
        st.error(f"حدث خطأ في حساب المؤشرات الفنية: {str(e)}")
        return data

def plot_technical_analysis(data, ticker):
    fig = go.Figure()
    
    try:
        # التحقق من وجود الأعمدة المطلوبة
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            st.error("بيانات الأسعار غير مكتملة للتحليل الفني")
            return fig
            
        # الرسم البياني للأسعار
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='أسعار الأسهم',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        
        # Bollinger Bands إذا كانت متوفرة
        if all(col in data.columns for col in ['BB_high', 'BB_low', 'BB_mid']):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_high'],
                line=dict(color='rgba(250, 0, 0, 0.5)',
                name='النطاق العلوي',
                fill=None
            )))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_low'],
                line=dict(color='rgba(0, 250, 0, 0.5)',
                name='النطاق السفلي',
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.1)'
            )))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_mid'],
                line=dict(color='blue', width=1),
                name='الخط الأوسط'
            ))
        
        # إعدادات التخطيط المحسنة
        fig.update_layout(
            title=f'التحليل الفني لسهم {ticker}',
            xaxis_title='التاريخ',
            yaxis_title='السعر',
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1 شهر", step="month", stepmode="backward"),
                    dict(count=3, label="3 أشهر", step="month", stepmode="backward"),
                    dict(count=6, label="6 أشهر", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
    except Exception as e:
        st.error(f"حدث خطأ في إنشاء الرسم البياني: {str(e)}")
    
    return fig

# ---------------------------------------------------
# قسم التنبؤ بالأسهم المحسن
# ---------------------------------------------------
def prepare_data_for_prediction(data):
    try:
        if data.empty or 'Close' not in data.columns:
            return pd.DataFrame(), pd.Series()
            
        # إنشاء متغيرات للتنبؤ مع تحسينات
        data['Price_Up'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=5).std()
        data = data.dropna()
        
        # تحديد الميزات والهدف
        features = data[['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'Volatility']]
        target = data['Price_Up']
        
        return features, target
    except Exception as e:
        st.error(f"حدث خطأ في تحضير بيانات التنبؤ: {str(e)}")
        return pd.DataFrame(), pd.Series()

def train_prediction_model(features, target):
    try:
        if features.empty or target.empty or len(features) < 30:
            return None, 0
            
        # تقسيم البيانات مع تحسينات
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, shuffle=False
        )
        
        # تدريب النموذج مع معلمات محسنة
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # تقييم النموذج
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        return model, mse
    except Exception as e:
        st.error(f"حدث خطأ في تدريب النموذج: {str(e)}")
        return None, 0

def predict_next_day(model, last_data):
    try:
        if model is None or last_data.empty:
            return 0.5
            
        required_cols = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'Volatility']
        if not all(col in last_data.index for col in required_cols):
            return 0.5
            
        # تحضير بيانات اليوم الأخير للتنبؤ
        last_features = last_data[required_cols].values.reshape(1, -1)
        prediction = model.predict(last_features)[0]
        return max(0, min(1, prediction))  # التأكد من أن التوقع بين 0 و 1
    except:
        return 0.5

# ---------------------------------------------------
# واجهة المستخدم المحسنة
# ---------------------------------------------------
def main():
    # شريط جانبي للإعدادات
    with st.sidebar:
        st.header('الإعدادات')
        
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input('رمز السهم (مثال: AAPL)', 'AAPL').strip().upper()
        with col2:
            days_back = st.slider('عدد الأيام للتحليل', 30, 365, 180)
        
        start_date = st.date_input('تاريخ البداية', datetime.now() - timedelta(days=days_back))
        end_date = st.date_input('تاريخ النهاية', datetime.now())
        
        st.markdown("---")
        st.markdown("""
        **مصادر البيانات:**
        - Yahoo Finance
        - Tiingo API
        - NewsAPI
        """)
        st.markdown("---")
        st.markdown("""
        **إعدادات التحليل:**
        - فترة RSI: 14 يوم
        - فترة Bollinger Bands: 20 يوم
        - فترة MACD: 12/26/9
        """)
    
    # تبويبات الواجهة
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 الصفحة الرئيسية",
        "📰 الأخبار العاجلة",
        "📊 التحليل الفني",
        "🔮 التنبؤ بالأسهم"
    ])
    
    with tab1:
        st.header("الأسهم الأكثر ارتفاعاً اليوم")
        with st.spinner('جاري تحميل بيانات الأسهم الصاعدة...'):
            gainers = get_top_gainers()
            if not gainers.empty:
                # تحسين عرض الجدول
                st.dataframe(
                    gainers.style
                    .highlight_max(axis=0, color='lightgreen')
                    .highlight_min(axis=0, color='#ffcccb')
                    .format({
                        'priceChange': '{:.2f}',
                        'priceChangePercent': '{:.2f}%',
                        'volume': '{:,.0f}'
                    })
                )
            else:
                st.warning("لا يمكن جلب بيانات الأسهم الصاعدة حالياً. يرجى المحاولة لاحقاً.")
        
        st.markdown("---")
        st.header("أهم المؤشرات الأمريكية")
        
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "Nasdaq": "^IXIC"
        }

        cols = st.columns(3)
        for i, (name, ticker_symbol) in enumerate(indices.items()):
            try:
                with st.spinner(f'جاري تحميل بيانات {name}...'):
                    data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
                    if not data.empty and 'Close' in data.columns and len(data['Close']) > 0:
                        close_prices = data['Close']
                        change = ((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) * 100
                        cols[i].metric(
                            label=name,
                            value=f"{close_prices.iloc[-1]:,.2f}",
                            delta=f"{change:.2f}%",
                            delta_color="normal" if change >= 0 else "inverse"
                        )
                        
                        # رسم سريع للتغير
                        fig, ax = plt.subplots(figsize=(4, 1))
                        ax.plot(close_prices, color='green' if change >= 0 else 'red', linewidth=1)
                        ax.axis('off')
                        cols[i].pyplot(fig, use_container_width=True)
                        plt.close()
                    else:
                        cols[i].metric(label=name, value="N/A", delta="N/A")
            except Exception as e:
                cols[i].metric(label=name, value="Error", delta=str(e))
    
    with tab2:
        st.header("آخر الأخبار المالية العاجلة")
        with st.spinner('جاري تحميل الأخبار...'):
            news = get_financial_news()
            
            if news:
                for i, article in enumerate(news[:5]):  # عرض 5 أخبار فقط لأغراض الأداء
                    with st.expander(f"{article.get('title', 'لا يوجد عنوان')}"):
                        st.markdown(f"""
                        **المصدر:** {article.get('source', {}).get('name', 'غير معروف')}  
                        **التاريخ:** {article.get('publishedAt', '')[:10]}  
                        **الوصف:** {article.get('description', 'لا يوجد وصف')}  
                        """)
                        if st.button("قراءة المزيد", key=f"news_{i}"):
                            st.markdown(f"[فتح المقال الأصلي]({article.get('url', '#')})")
            else:
                st.warning("لا يمكن جلب الأخبار حالياً. يرجى التحقق من اتصال الإنترنت أو مفتاح API.")
    
    with tab3:
        st.header("التحليل الفني المتقدم")
        
        if ticker:
            try:
                with st.spinner('جاري تحميل بيانات السهم...'):
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if not data.empty:
                        # حساب المؤشرات الفنية
                        with st.spinner('جاري حساب المؤشرات الفنية...'):
                            data = calculate_technical_indicators(data)
                        
                        # عرض الرسم البياني
                        fig = plot_technical_analysis(data, ticker)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # عرض المؤشرات الفنية في بطاقات
                        st.subheader("قراءات المؤشرات الفنية")
                        cols = st.columns(4)
                        
                        # RSI
                        if 'RSI' in data.columns:
                            current_rsi = data['RSI'].iloc[-1]
                            rsi_color = "red" if current_rsi > 70 else "green" if current_rsi < 30 else "blue"
                            rsi_status = "مشترى زائد (فوق 70)" if current_rsi > 70 else "مبيع زائد (تحت 30)" if current_rsi < 30 else "محايد"
                            cols[0].metric(
                                "RSI (14 يوم)", 
                                f"{current_rsi:.2f}", 
                                rsi_status,
                                delta_color="off"
                            )
                            cols[0].progress(min(100, int(current_rsi)))
                        
                        # MACD
                        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                            macd_diff = data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]
                            macd_status = "إشارة شراء" if macd_diff > 0 else "إشارة بيع"
                            cols[1].metric(
                                "MACD", 
                                f"{macd_diff:.4f}", 
                                macd_status,
                                delta_color="normal" if macd_diff > 0 else "inverse"
                            )
                        
                        # Bollinger Bands
                        if all(col in data.columns for col in ['Close', 'BB_high', 'BB_low']):
                            current_price = data['Close'].iloc[-1]
                            bb_high = data['BB_high'].iloc[-1]
                            bb_low = data['BB_low'].iloc[-1]
                            bb_status = "قرب المقاومة" if current_price > bb_high * 0.95 else "قرب الدعم" if current_price < bb_low * 1.05 else "في النطاق"
                            bb_percent = ((current_price - bb_low) / (bb_high - bb_low)) * 100
                            cols[2].metric(
                                "Bollinger Bands", 
                                bb_status,
                                delta=f"{bb_percent:.1f}%",
                                delta_color="off"
                            )
                            cols[2].progress(int(bb_percent))
                        
                        # حجم التداول
                        if 'Volume' in data.columns:
                            vol_mean = data['Volume'].mean()
                            if vol_mean > 0:
                                vol_change = ((data['Volume'].iloc[-1] - vol_mean) / vol_mean) * 100
                                cols[3].metric(
                                    "حجم التداول", 
                                    f"{data['Volume'].iloc[-1]/1e6:.2f}M", 
                                    f"{vol_change:.2f}%",
                                    delta_color="normal" if vol_change >= 0 else "inverse"
                                )
                        
                        # عرض بيانات الأسهم مع خيارات تصفية
                        st.subheader("بيانات الأسهم التاريخية")
                        st.dataframe(
                            data.tail(20).style
                            .format({
                                'Open': '{:.2f}',
                                'High': '{:.2f}',
                                'Low': '{:.2f}',
                                'Close': '{:.2f}',
                                'Volume': '{:,.0f}',
                                'RSI': '{:.2f}',
                                'MACD': '{:.4f}'
                            })
                            .applymap(lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red', 
                                    subset=['MACD', 'MACD_hist'])
                        )
                    else:
                        st.error(f"لا توجد بيانات متاحة للسهم {ticker}. يرجى التحقق من الرمز.")
            except Exception as e:
                st.error(f"حدث خطأ في تحميل بيانات السهم: {str(e)}")
    
    with tab4:
        st.header("تنبؤ حركة السهم")
        
        if ticker:
            try:
                with st.spinner('جاري تحميل بيانات السهم...'):
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if not data.empty and len(data) > 30:
                        # حساب المؤشرات الفنية
                        with st.spinner('جاري حساب المؤشرات...'):
                            data = calculate_technical_indicators(data)
                        
                        # تحضير البيانات للتنبؤ
                        features, target = prepare_data_for_prediction(data)
                        
                        if not features.empty and not target.empty:
                            # تدريب النموذج
                            with st.spinner('جاري تدريب نموذج التنبؤ...'):
                                model, mse = train_prediction_model(features, target)
                            
                            if model:
                                st.success(f"تم تدريب النموذج بنجاح (خطأ مربع متوسط: {mse:.4f})")
                                
                                # التنبؤ لليوم التالي
                                last_data = data.iloc[-1]
                                prediction = predict_next_day(model, last_data)
                                
                                # عرض النتائج في بطاقات
                                st.subheader("توقعات حركة السهم")
                                cols = st.columns(3)
                                
                                cols[0].metric(
                                    "السعر الحالي", 
                                    f"{last_data.get('Close', 0):.2f}",
                                    delta=f"{(last_data.get('Close', 0) - data['Close'].iloc[-2]):.2f}",
                                    delta_color="normal" if last_data.get('Close', 0) >= data['Close'].iloc[-2] else "inverse"
                                )
                                
                                cols[1].metric(
                                    "ثقة النموذج", 
                                    f"{abs(prediction-0.5)*200:.1f}%",
                                    delta_color="off"
                                )
                                
                                if prediction > 0.6:
                                    cols[2].metric(
                                        "التوقع", 
                                        "ارتفاع محتمل", 
                                        delta="↑↑ قوي",
                                        delta_color="normal"
                                    )
                                elif prediction > 0.55:
                                    cols[2].metric(
                                        "التوقع", 
                                        "ارتفاع طفيف", 
                                        delta="↑ محتمل",
                                        delta_color="normal"
                                    )
                                elif prediction < 0.4:
                                    cols[2].metric(
                                        "التوقع", 
                                        "انخفاض محتمل", 
                                        delta="↓↓ قوي",
                                        delta_color="inverse"
                                    )
                                elif prediction < 0.45:
                                    cols[2].metric(
                                        "التوقع", 
                                        "انخفاض طفيف", 
                                        delta="↓ محتمل",
                                        delta_color="inverse"
                                    )
                                else:
                                    cols[2].metric(
                                        "التوقع", 
                                        "مستقر", 
                                        delta="→ محايد",
                                        delta_color="off"
                                    )
                                
                                # تفسير النتائج
                                with st.expander("كيفية تفسير التوقعات"):
                                    st.markdown("""
                                    **مستويات الثقة:**
                                    - **أعلى من 60%:** إشارة شراء قوية (احتمال ارتفاع السعر)
                                    - **بين 55%-60%:** إشارة شراء متوسطة
                                    - **بين 45%-55%:** سوق متقلب/مستقر
                                    - **بين 40%-45%:** إشارة بيع متوسطة
                                    - **أقل من 40%:** إشارة بيع قوية (احتمال انخفاض السعر)
                                    
                                    *ملاحظة: هذه التوقعات تعتمد على نماذج التعلم الآلي ولا تضمن الدقة الكاملة.*
                                    """)
                                
                                # أهمية الميزات
                                st.subheader("أهم العوامل في التنبؤ")
                                feature_importance = pd.DataFrame({
                                    'الميزة': features.columns,
                                    'الأهمية': model.feature_importances_
                                }).sort_values('الأهمية', ascending=False)
                                
                                fig = go.Figure(go.Bar(
                                    x=feature_importance['الأهمية'],
                                    y=feature_importance['الميزة'],
                                    orientation='h',
                                    marker_color='skyblue'
                                ))
                                fig.update_layout(
                                    title="أهمية كل ميزة في التنبؤ",
                                    xaxis_title="مستوى الأهمية",
                                    yaxis_title="الميزة",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # عرض بيانات التنبؤ
                                st.subheader("آخر 10 تنبؤات")
                                predictions = model.predict(features.tail(10))
                                actuals = target.tail(10).values
                                
                                pred_df = pd.DataFrame({
                                    'التاريخ': features.tail(10).index,
                                    'التنبؤ': predictions,
                                    'النتيجة الفعلية': actuals,
                                    'الصحة': np.where((predictions > 0.5) == (actuals > 0.5), 'صحيح', 'خطأ')
                                })
                                
                                st.dataframe(
                                    pred_df.style
                                    .applymap(lambda x: 'background-color: lightgreen' if x == 'صحيح' else 'background-color: #ffcccb', 
                                            subset=['الصحة'])
                                    .format({
                                        'التنبؤ': '{:.2%}',
                                        'النتيجة الفعلية': '{:.0f}'
                                    })
                                )
                        else:
                            st.error("لا توجد بيانات كافية للتنبؤ.")
                    else:
                        st.error("لا توجد بيانات كافية للتنبؤ. يحتاج النموذج إلى بيانات 30 يوم على الأقل.")
            except Exception as e:
                st.error(f"حدث خطأ في عملية التنبؤ: {str(e)}")

if __name__ == "__main__":
    main()
