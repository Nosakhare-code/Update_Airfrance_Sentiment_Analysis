import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import time
from transformers import pipeline
import torch

# Page configuration
st.set_page_config(
    page_title="Airline Customer Sentiment Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding-top: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .recommendation-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .twitter-comment {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .negative-comment { border-left: 5px solid #EF4444; }
    .positive-comment { border-left: 5px solid #10B981; }
    .neutral-comment { border-left: 5px solid #F59E0B; }
</style>
""", unsafe_allow_html=True)

# Initialize Hugging Face sentiment pipeline
@st.cache_resource
def load_sentiment_model():
    """Load the sentiment analysis model from Hugging Face"""
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Use CPU
        )
        return sentiment_pipeline
    except Exception as e:
        st.warning(f"Could not load model: {e}. Using simulated analysis.")
        return None

# Initialize session state
if 'sentiment_model' not in st.session_state:
    st.session_state.sentiment_model = load_sentiment_model()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'live_tweets' not in st.session_state:
    st.session_state.live_tweets = []

# Twitter simulation function
def fetch_twitter_comments_simulated(search_term="AirFrance", count=10):
    """Simulate fetching tweets"""
    simulated_tweets = [
        {
            "text": "@AirFrance worst customer service ever! Been on hold for 2 hours #airfrance",
            "username": "traveler_123",
            "timestamp": datetime.now() - timedelta(hours=1),
            "likes": 15,
            "retweets": 3
        },
        {
            "text": "Air France handled my flight cancellation professionally. Thank you!",
            "username": "happy_customer",
            "timestamp": datetime.now() - timedelta(hours=2),
            "likes": 42,
            "retweets": 12
        },
        {
            "text": "Trying to change my Air France ticket and keep getting disconnected. Very frustrating!",
            "username": "frustrated_flyer",
            "timestamp": datetime.now() - timedelta(hours=3),
            "likes": 8,
            "retweets": 2
        },
        {
            "text": "Air France agent Maria was extremely helpful with my visa issue. Great service!",
            "username": "grateful_passenger",
            "timestamp": datetime.now() - timedelta(hours=4),
            "likes": 56,
            "retweets": 18
        },
        {
            "text": "Another hour waiting for Air France customer service. This is unacceptable.",
            "username": "waiting_forever",
            "timestamp": datetime.now() - timedelta(hours=5),
            "likes": 23,
            "retweets": 7
        },
        {
            "text": "Booked Air France business class. Excellent experience from start to finish!",
            "username": "business_traveler",
            "timestamp": datetime.now() - timedelta(hours=6),
            "likes": 89,
            "retweets": 24
        },
        {
            "text": "Air France lost my luggage and won't respond to emails. Terrible service!",
            "username": "angry_customer",
            "timestamp": datetime.now() - timedelta(hours=7),
            "likes": 31,
            "retweets": 9
        },
        {
            "text": "Great in-flight entertainment on Air France. Loved the movie selection!",
            "username": "entertainment_lover",
            "timestamp": datetime.now() - timedelta(hours=8),
            "likes": 47,
            "retweets": 14
        }
    ]
    return simulated_tweets[:count]

def analyze_sentiment_live(text):
    """Analyze sentiment of text using Hugging Face model"""
    if st.session_state.sentiment_model:
        try:
            result = st.session_state.sentiment_model(text[:512])[0]
            label = result['label']
            score = result['score']
            
            if label == "POSITIVE":
                return "Positive", score
            elif label == "NEGATIVE":
                return "Negative", score
            else:
                return "Neutral", score
        except Exception as e:
            st.error(f"Error in sentiment analysis: {e}")
            return "Neutral", 0.5
    else:
        # Fallback to keyword-based analysis
        negative_keywords = ['terrible', 'worst', 'frustrating', 'unacceptable', 'disconnected', 'lost', 'angry']
        positive_keywords = ['excellent', 'great', 'helpful', 'professional', 'thank', 'loved', 'good']
        
        text_lower = text.lower()
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        
        if negative_count > positive_count:
            return "Negative", 0.8
        elif positive_count > negative_count:
            return "Positive", 0.8
        else:
            return "Neutral", 0.5

def clean_text(text):
    """Clean text for analysis"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# Title and introduction
st.markdown('<h1 class="main-header">✈️ Airline Live Customer Sentiment Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
### Real-time Sentiment Analysis & Economic Impact Assessment
*Project done by Emmanuel Noskhare Asowata* \n
*Contact --> noskhareasowata94@gmail.com* \n
*Quantitative & Applied Economist* \n
*Powered by Hugging Face Transformers & Streamlit*.
""")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Air_France_Logo.svg/2560px-Air_France_Logo.svg.png", 
             use_container_width=True)
    
    st.markdown("### ⚙️ Analysis Controls")
    
    # Real-time analysis controls
    st.markdown("**Live Analysis**")
    if st.button("🔄 Fetch New Tweets", use_container_width=True):
        with st.spinner("Fetching latest tweets..."):
            new_tweets = fetch_twitter_comments_simulated()
            for tweet in new_tweets:
                sentiment, score = analyze_sentiment_live(tweet["text"])
                tweet["sentiment"] = sentiment
                tweet["sentiment_score"] = score
                tweet["clean_text"] = clean_text(tweet["text"])
            st.session_state.live_tweets = new_tweets + st.session_state.live_tweets[:20]
            st.success(f"Analyzed {len(new_tweets)} new tweets!")
    
    # Manual text input for analysis
    st.markdown("**Analyze Custom Text**")
    custom_text = st.text_area("Enter text to analyze:", height=100)
    if st.button("🔍 Analyze Sentiment", use_container_width=True) and custom_text:
        with st.spinner("Analyzing sentiment..."):
            sentiment, score = analyze_sentiment_live(custom_text)
            st.session_state.analysis_history.append({
                "text": custom_text[:100] + "..." if len(custom_text) > 100 else custom_text,
                "sentiment": sentiment,
                "score": score,
                "timestamp": datetime.now()
            })
            
            # Display result
            sentiment_color = {
                "Negative": "🔴",
                "Neutral": "🟡",
                "Positive": "🟢"
            }[sentiment]
            
            st.markdown(f"""
            <div style="background: {'#FEE2E2' if sentiment == 'Negative' else '#FEF3C7' if sentiment == 'Neutral' else '#D1FAE5'}; 
                        padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h4>{sentiment_color} {sentiment} ({score:.2f})</h4>
                <p>{custom_text}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Economic parameters
    st.markdown("---")
    st.markdown("### 💰 Economic Parameters")
    avg_ticket_price = st.slider("Average Ticket Price (€)", 500, 2000, 800, 50)
    churn_rate = st.slider("Negative Sentiment Churn Rate (%)", 5, 50, 20, 5) / 100
    retention_value = st.slider("Customer Lifetime Value (€)", 2000, 20000, 5000, 500)
    
    # Model information
    st.markdown("---")
    with st.expander("🤖 Model Information"):
        if st.session_state.sentiment_model:
            st.success("✓ Hugging Face Model Loaded")
            st.info("Model: distilbert-base-uncased-finetuned-sst-2-english")
        else:
            st.warning("⚠️ Using simulated analysis")
        st.caption("Transformers provide state-of-the-art sentiment analysis")

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Case Study", "💡 Recommendations", "📈 Live Analysis"])

with tab1:
    # Load synthetic data
    @st.cache_data
    def load_base_data():
        complaint_templates = [
            "@AirFrance been on hold for 2 hours trying to change my flight. Customer service is terrible!",
            "Air France cancelled my flight due to visa issues and now charging $400 to rebook. Unacceptable!",
            "Every time I call @AirFrance I get a different agent and have to explain everything again. So frustrating!",
            "Air France lost my luggage and no one answers the phone. Worst customer service ever!",
            "Trying to get a refund from @AirFrance for 3 months. They keep transferring me between departments.",
            "Air France agent couldn't understand my English properly. Communication issues with customer service.",
            "Booked with Air France but had to cancel due to visa delay. They want huge change fees!",
            "@AirFrance please help! I've been trying to reach customer service for days about my booking.",
            "Air France flight cancelled, no alternatives provided. Stranded at airport!",
            "The @AirFrance call center disconnected me twice after waiting 45 minutes each time.",
            "Positive: Air France resolved my issue quickly when I finally got through to the right department.",
            "Neutral: Contacted Air France about flight change, waiting to hear back.",
            "Air France premium customer but treated like everyone else when issues arise. Disappointing.",
            "Agent was helpful but system wouldn't let them make the change. Air France needs better tech.",
            "Had to explain my visa situation 3 times to different agents at @AirFrance. No continuity!",
            "Air France website shows one price, agent quotes another. Inconsistent information.",
            "Flight to Nigeria cancelled, Air France won't let me change to Accra without huge fees.",
            "Good service from Air France agent Maria today. She understood my urgent situation.",
            "Terrible experience with Air France customer service. Will fly Emirates next time.",
            "@AirFrance please train your agents better. Language barrier is a real problem."
        ]
        
        general_df = pd.DataFrame({
            "date": pd.date_range(start="2023-08-01", periods=len(complaint_templates), freq="D"),
            "text": complaint_templates,
            "source": "General Twitter"
        })
        
        # Add sentiment from live model
        sentiments = []
        scores = []
        for text in complaint_templates:
            sentiment, score = analyze_sentiment_live(text)
            sentiments.append(sentiment)
            scores.append(score)
        
        general_df["sentiment_label"] = sentiments
        general_df["sentiment_score"] = scores
        
        return general_df
    
    base_df = load_base_data()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tweets = len(base_df) + len(st.session_state.live_tweets)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Tweets Analyzed</h4>
            <h2>{total_tweets:,}</h2>
            <p>{len(st.session_state.live_tweets)} live + {len(base_df)} historical</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        all_negative = sum(1 for x in base_df["sentiment_label"] if x == "Negative") + \
                      sum(1 for x in st.session_state.live_tweets if x.get("sentiment") == "Negative")
        negative_pct = (all_negative / total_tweets * 100) if total_tweets > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>Negative Sentiment</h4>
            <h2>{negative_pct:.1f}%</h2>
            <p>{all_negative} negative tweets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_score = np.mean(list(base_df["sentiment_score"]) + 
                           [t.get("sentiment_score", 0.5) for t in st.session_state.live_tweets])
        st.markdown(f"""
        <div class="metric-card">
            <h4>Avg Sentiment Score</h4>
            <h2>{avg_score:.2f}</h2>
            <p>Higher is better (0-1 scale)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        potential_loss = all_negative * churn_rate * retention_value
        st.markdown(f"""
        <div class="metric-card">
            <h4>Revenue at Risk</h4>
            <h2>€{potential_loss:,.0f}</h2>
            <p>Based on current analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        sentiment_counts = pd.Series([x for x in base_df["sentiment_label"]] + 
                                    [t.get("sentiment", "Neutral") for t in st.session_state.live_tweets]).value_counts()
        fig1 = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Overall Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={
                "Negative": "#EF4444",
                "Neutral": "#F59E0B",
                "Positive": "#10B981"
            }
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Sentiment over time
        dates = pd.date_range(start='2023-08-01', periods=30, freq='D')
        sentiment_values = np.random.choice([-1, 0, 1], 30, p=[0.6, 0.2, 0.2])
        cumulative = np.cumsum(sentiment_values)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dates, y=cumulative,
            mode='lines',
            name='Sentiment Trend',
            line=dict(color='#2563EB', width=3)
        ))
        fig2.update_layout(
            title="Sentiment Trend (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Cumulative Sentiment Score",
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown('<h2 class="sub-header">🎯 The Agent Continuity Problem</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### The Case Study: Visa Issues & Agent Discontinuity
        
        **The Problem:**
        - Customer books flight to Nigeria
        - E-visa approval delayed by immigration
        - Needs to change destination to Accra, Ghana
        - Willing to pay extra fees
        
        **Customer Experience Breakdown:**
        1. **Call 1:** Agent doesn't understand English clearly
        2. **Call 2:** Helpful agent but system issues
        3. **Call 3:** New agent, must repeat entire story
        4. **Call 4:** Different agent, different pricing information
        5. **Call 5:** Finally reaches helpful agent again
        
        **Time Lost:** 5+ hours over multiple days
        **Customer Frustration:** Extremely high
        **Business Impact:** Risk of losing customer forever
        """)
        
        # Journey visualization
        journey_data = pd.DataFrame({
            'Step': ['Call 1', 'Call 2', 'Call 3', 'Call 4', 'Call 5'],
            'Sentiment': ['Negative', 'Positive', 'Negative', 'Negative', 'Positive'],
            'Time (min)': [45, 30, 60, 40, 20],
            'Outcome': ['No Progress', 'Partial Progress', 'Start Over', 'Confusion', 'Resolution']
        })
        
        fig3 = px.bar(journey_data, x='Step', y='Time (min)', color='Sentiment',
                     color_discrete_map={'Negative': '#EF4444', 'Positive': '#10B981'},
                     title="Customer Journey Timeline")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Economic Impact Analysis
        
        **Direct Costs:**
        - Call center time: €150
        - Agent training inefficiency: €200
        - System resources: €50
        
        **Indirect Costs:**
        - Customer acquisition cost to replace: €500
        - Lifetime value at risk: €5,000
        - Negative word-of-mouth: €2,000+
        
        **Total Cost of This Single Incident:**
        """)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
            <h3>€{150+200+50+500+5000+2000:,.0f}+</h3>
            <p>Potential loss from one customer experience</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Scaled Impact:**
        - 100 similar cases/month = €790,000 monthly
        - Annual impact = €9.5+ million
        """)

with tab3:
    st.markdown('<h2 class="sub-header">🚀 Data-Driven Recommendations</h2>', unsafe_allow_html=True)
    
    # ROI Calculator
    st.markdown("### 💰 ROI Calculator for Proposed Solutions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        solution_cost = st.number_input("Solution Cost (€)", value=200000, step=50000)
    with col2:
        expected_improvement = st.slider("Expected CSAT Improvement (%)", 5, 50, 20)
    with col3:
        customers_affected = st.number_input("Customers Affected Monthly", value=10000)
    
    # Calculate ROI
    value_per_point = customers_affected * retention_value * 0.01
    annual_value = value_per_point * expected_improvement * 12
    roi = ((annual_value - solution_cost) / solution_cost) * 100
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
        <h3>Projected Annual ROI: {roi:.0f}%</h3>
        <p>Annual Value: €{annual_value:,.0f} | Investment: €{solution_cost:,.0f}</p>
        <p>Net Gain: €{annual_value - solution_cost:,.0f} per year</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 🎯 Specific Recommendations")
    
    recommendations = [
        {
            "title": "Agent Continuity System",
            "description": "Implement case history tracking and customer-agent pairing",
            "cost": "€150,000",
            "roi": "300%",
            "timeline": "3 months"
        },
        {
            "title": "Language Proficiency Program",
            "description": "Mandatory English certification for customer-facing agents",
            "cost": "€80,000",
            "roi": "250%",
            "timeline": "2 months"
        },
        {
            "title": "Flexible Visa Policy",
            "description": "Streamlined process for visa-related changes",
            "cost": "€50,000",
            "roi": "400%",
            "timeline": "1 month"
        },
        {
            "title": "Real-time Sentiment Monitoring",
            "description": "Live dashboard for customer sentiment tracking",
            "cost": "€100,000",
            "roi": "200%",
            "timeline": "4 months"
        }
    ]
    
    for i, rec in enumerate(recommendations):
        st.markdown(f"""
        <div class="recommendation-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h5>#{i+1} {rec['title']}</h5>
                    <p>{rec['description']}</p>
                </div>
                <div style="text-align: right;">
                    <h6>Cost: {rec['cost']}</h6>
                    <h6>ROI: {rec['roi']}</h6>
                    <h6>Timeline: {rec['timeline']}</h6>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown('<h2 class="sub-header">📈 Live Twitter Analysis</h2>', unsafe_allow_html=True)
    
    # Real-time analysis controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("Search Term", value="AirFrance")
    with col2:
        tweet_count = st.selectbox("Tweet Count", [5, 10, 20, 50], index=1)
    with col3:
        st.markdown("")
        st.markdown("")
        if st.button("🚀 Run Live Analysis", type="primary"):
            with st.spinner(f"Fetching and analyzing {tweet_count} tweets..."):
                time.sleep(2)
                
                tweets = fetch_twitter_comments_simulated(search_term, tweet_count)
                analyzed_tweets = []
                
                for tweet in tweets:
                    sentiment, score = analyze_sentiment_live(tweet["text"])
                    tweet["sentiment"] = sentiment
                    tweet["sentiment_score"] = score
                    tweet["clean_text"] = clean_text(tweet["text"])
                    analyzed_tweets.append(tweet)
                
                st.session_state.live_tweets = analyzed_tweets + st.session_state.live_tweets
                st.success(f"Analyzed {len(analyzed_tweets)} tweets!")
    
    # Display live tweets
    if st.session_state.live_tweets:
        st.markdown(f"### 📊 Recently Analyzed Tweets ({len(st.session_state.live_tweets)} total)")
        
        # Sentiment summary
        sentiment_summary = pd.Series([t["sentiment"] for t in st.session_state.live_tweets]).value_counts()
        
        col1, col2, col3 = st.columns(3)
        for sentiment, count in sentiment_summary.items():
            with col1 if sentiment == "Negative" else col2 if sentiment == "Neutral" else col3:
                color = "#EF4444" if sentiment == "Negative" else "#F59E0B" if sentiment == "Neutral" else "#10B981"
                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                    <h4>{sentiment}</h4>
                    <h2>{count}</h2>
                    <p>{(count/len(st.session_state.live_tweets)*100):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display tweets
        st.markdown("### 📝 Tweet Details")
        for tweet in st.session_state.live_tweets[:10]:
            sentiment_class = f"{tweet['sentiment'].lower()}-comment"
            sentiment_emoji = "🔴" if tweet["sentiment"] == "Negative" else "🟡" if tweet["sentiment"] == "Neutral" else "🟢"
            
            st.markdown(f"""
            <div class="twitter-comment {sentiment_class}">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <div>
                        <strong>@{tweet['username']}</strong>
                        <span style="margin-left: 10px; opacity: 0.8;">
                            {tweet['timestamp'].strftime('%Y-%m-%d %H:%M')}
                        </span>
                    </div>
                    <div>
                        {sentiment_emoji} <strong>{tweet['sentiment']}</strong> ({tweet['sentiment_score']:.2f})
                    </div>
                </div>
                <p style="margin: 0;">{tweet['text']}</p>
                <div style="display: flex; gap: 15px; margin-top: 10px; opacity: 0.8;">
                    <span>❤️ {tweet['likes']}</span>
                    <span>🔄 {tweet['retweets']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Export data
        if st.button("📥 Export Analysis Results"):
            export_df = pd.DataFrame(st.session_state.live_tweets)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="live_twitter_analysis.csv",
                mime="text/csv"
            )
    else:
        st.info("👈 Click 'Fetch New Tweets' in the sidebar or 'Run Live Analysis' above to start!")

# Analysis History
if st.session_state.analysis_history:
    with st.expander("📋 Analysis History"):
        for analysis in reversed(st.session_state.analysis_history[-5:]):
            st.markdown(f"""
            **{analysis['timestamp'].strftime('%H:%M')}** - {analysis['sentiment']} ({analysis['score']:.2f})
            > {analysis['text']}
            ---
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 2rem;">
    <h4>🌐 Live Deployment Information</h4>
    <p>This dashboard is deployed on <strong>Streamlit Community Cloud</strong> with live Hugging Face integration.</p>
    <p>
        <strong>Models Used:</strong> distilbert-base-uncased-finetuned-sst-2-english (Hugging Face) |
        <strong>Updates:</strong> Real-time sentiment analysis every refresh |
        <strong>Data:</strong> Historical + Simulated
    </p>
</div>
""", unsafe_allow_html=True)
