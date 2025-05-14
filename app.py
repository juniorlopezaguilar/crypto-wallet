import streamlit as st
import openai
import requests
import pandas as pd
from openai import OpenAI
from datetime import datetime, timedelta

st.title("Chat about Crypto Wallets")

# Get OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Ask me about crypto wallets!"}]

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def get_openai_response(prompt):
    """
    Gets a minimal response from a potentially less expensive OpenAI model.
    """
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",  # A generally efficient and capable model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,  # Limit the output tokens
            temperature=0.7, # Adjust for more deterministic (lower) or creative (higher) responses
            n=1             # Request only one response
        )
        return completion.choices[0].message.content
    except openai.OpenAIError as e:
        return f"An error occurred: {e}"


if prompt := st.chat_input("Ask about crypto wallets!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        openai_response = get_openai_response(prompt)
        full_response = openai_response


        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


@st.cache_data
def get_btc_price_history():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "7",  # past 7 days
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    if response.ok:
        data = response.json()
        prices = [
            {
                "Date": datetime.fromtimestamp(price[0] / 1000).strftime('%Y-%m-%d'),
                "Price (USD)": price[1]
            }
            for price in data["prices"]
        ]
        return pd.DataFrame(prices)
    else:
        return pd.DataFrame(columns=["Date", "Price (USD)"])

st.title("📈 Bitcoin Price Trend (Last 7 Days)")
df = get_btc_price_history()

if df.empty:
    st.warning("No data available from CoinGecko.")
else:
    st.dataframe(df)
    st.line_chart(df.set_index("Date"))
