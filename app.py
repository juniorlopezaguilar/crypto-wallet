import streamlit as st
import openai
import requests
import pandas as pd
from openai import OpenAI
from datetime import datetime, timedelta

st.header("Demo Crypto chat with multiple agents")

# ------------ Get OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ------------ Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¡Hola! Pregúntame sobre billeteras de criptomonedas y monedas digitales!"}]

# ------------ Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

@st.cache_data
def get_price_history(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
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


class CoinIdentifierAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.coin_mapping = {
            "btc": "bitcoin",
            "bitcoin": "bitcoin",
            "eth": "ethereum",
            "ethereum": "ethereum",
            # Add more mappings as needed
        }

    def identify_coin(self, prompt):
        """Identifies if the user is asking about a specific digital coin and returns its CoinGecko ID."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that identifies if the user is asking about a specific digital coin. If so, return the lowercase CoinGecko ID of the coin. If not, or if you are unsure, return 'None'."},
                    {"role": "user", "content": f"Is the user asking about a specific digital coin in this prompt: '{prompt}'? If yes, which one (return its CoinGecko ID)?"}
                ],
                max_tokens=30,
                temperature=0.3
            )
            identification = response.choices[0].message.content.strip().lower()
            if identification and identification != "none":
                return self.coin_mapping.get(identification, None)
            return None
        except openai.OpenAIError as e:
            st.error(f"Error identifying digital coin: {e}")
            return None

class CryptoInfoAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_general_response(self, prompt):
        """Gets a general response from OpenAI."""
        try:
            client = OpenAI(api_key=self.api_key)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
                n=1
            )
            return completion.choices[0].message.content
        except openai.OpenAIError as e:
            return f"An error occurred: {e}"

    def display_price_history(self, coin_id):
        """Fetches and displays the price history dataframe."""
        df = get_price_history(coin_id)
        if not df.empty:
            st.dataframe(df)
            return True
        else:
            st.warning(f"Could not retrieve price history for {coin_id.capitalize()}.")
            return False

class TranslatorAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def translate_to_english(self, text):
        """Translates the given text to English."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates the user's input from Spanish to English."},
                    {"role": "user", "content": f"Translate the following Spanish text to English: '{text}'"}
                ],
                max_tokens=50,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except openai.OpenAIError as e:
            st.error(f"Error during translation: {e}")
            return text  # Return original text in case of error

class IntentRecognizerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def identify_price_history_intent(self, prompt):
        """Identifies if the user is asking about price history, trends, or fluctuations."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines if the user is asking about the price history, trends, or fluctuations of a cryptocurrency. Return 'history' if the intent is related to price history, trends, or fluctuations, otherwise return 'general'."},
                    {"role": "user", "content": f"Determine the intent of this prompt: '{prompt}'"}
                ],
                max_tokens=20,
                temperature=0.2  # Keep it focused
            )
            intent = response.choices[0].message.content.strip().lower()
            return intent
        except openai.OpenAIError as e:
            st.error(f"Error recognizing intent: {e}")
            return "general"

# ------------ Initialize Agents
coin_identifier_agent = CoinIdentifierAgent(openai.api_key)
crypto_info_agent = CryptoInfoAgent(openai.api_key)
translator_agent = TranslatorAgent(openai.api_key)
intent_recognizer_agent = IntentRecognizerAgent(openai.api_key)

# ------------ Main Function
if prompt := st.chat_input("Ask about crypto wallets, prices, history, trends (or ask in Spanish!)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Check if the prompt is likely in Spanish
        if any(char in "áéíóúñ¿¡" for char in prompt):
            english_prompt = translator_agent.translate_to_english(prompt)
            st.info(f"Translated from Spanish: '{prompt}' to English: '{english_prompt}'")
        else:
            english_prompt = prompt

        identified_coin = coin_identifier_agent.identify_coin(english_prompt)
        intent = intent_recognizer_agent.identify_price_history_intent(english_prompt)

        if identified_coin and "price" in english_prompt.lower(): # Prioritize simple price requests
            df = get_price_history(identified_coin)
            if df.empty:
                full_response = f"Sorry, I couldn't retrieve the price data for {identified_coin.capitalize()} right now."
            else:
                latest_price = df.iloc[-1]
                full_response = f"The latest price of {identified_coin.capitalize()} on {latest_price['Date']} is ${latest_price['Price (USD)']:.2f}."
                st.dataframe(df)
        elif identified_coin and intent == "history":
            if not crypto_info_agent.display_price_history(identified_coin):
                full_response = f"Sorry, I couldn't retrieve the price history for {identified_coin.capitalize()} right now."
        else:
            openai_response = crypto_info_agent.get_general_response(english_prompt)
            full_response = openai_response

        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response})