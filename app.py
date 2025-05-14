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

if "agent_interactions" not in st.session_state:
    st.session_state["agent_interactions"] = []

def record_interaction(agent_name, description):
    st.session_state["agent_interactions"].append({"agent": agent_name, "description": description})

def display_agent_interactions():
    with st.sidebar:
        st.subheader("Agent Interactions")
        if st.session_state["agent_interactions"]:
            for i, interaction in enumerate(st.session_state["agent_interactions"]):
                st.markdown(f"**{interaction['agent']} says:** {interaction['description']}")
                if i < len(st.session_state["agent_interactions"]) - 1:
                    st.divider()
        else:
            st.info("No agent interactions yet for this turn.")

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
        self.name = "Coin Identifier Agent"

    def identify_coin(self, prompt):
        """Identifies if the user is asking about a specific digital coin and returns its CoinGecko ID."""
        record_interaction(self.name, f"Analyzing user prompt: '{prompt}' to identify a specific cryptocurrency.")
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
            record_interaction(self.name, f"OpenAI responded with: '{identification}'.")
            if identification and identification != "none":
                coingecko_id = self.coin_mapping.get(identification, None)
                if coingecko_id:
                    record_interaction(self.name, f"Identified cryptocurrency: '{identification}', CoinGecko ID: '{coingecko_id}'.")
                    return coingecko_id
                else:
                    record_interaction(self.name, f"Identified '{identification}' but no CoinGecko ID mapping found.")
                    return None
            else:
                record_interaction(self.name, "No specific cryptocurrency identified in the prompt.")
                return None
        except openai.OpenAIError as e:
            error_message = f"Error identifying digital coin: {e}"
            record_interaction(self.name, error_message)
            st.error(error_message)
            return None

class CryptoInfoAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.name = "Crypto Information Agent"

    def get_general_response(self, prompt):
        """Gets a general response from OpenAI."""
        record_interaction(self.name, f"Received user prompt (English): '{prompt}'. Requesting general information from OpenAI.")
        try:
            client = OpenAI(api_key=self.api_key)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
                n=1
            )
            response_content = completion.choices[0].message.content
            record_interaction(self.name, f"OpenAI responded with general information: '{response_content}'.")
            return response_content
        except openai.OpenAIError as e:
            error_message = f"An error occurred while getting general information: {e}"
            record_interaction(self.name, error_message)
            return error_message

    def display_price_history(self, coin_id):
        """Fetches and displays the price history dataframe."""
        record_interaction(self.name, f"Fetching price history for CoinGecko ID: '{coin_id}'.")
        df = get_price_history(coin_id)
        if not df.empty:
            record_interaction(self.name, f"Successfully retrieved price history for '{coin_id}'. Displaying data.")
            st.dataframe(df)
            return True
        else:
            warning_message = f"Could not retrieve price history for {coin_id.capitalize()}."
            record_interaction(self.name, warning_message)
            st.warning(warning_message)
            return False

class TranslatorAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.name = "Translator Agent"

    def translate_to_english(self, text):
        """Translates the given text to English."""
        record_interaction(self.name, f"Received Spanish text: '{text}'. Requesting translation to English.")
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
            translated_text = response.choices[0].message.content.strip()
            record_interaction(self.name, f"Translated to English: '{translated_text}'.")
            return translated_text
        except openai.OpenAIError as e:
            error_message = f"Error during translation: {e}"
            record_interaction(self.name, error_message)
            st.error(error_message)
            return text  # Return original text in case of error

class IntentRecognizerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.name = "Intent Recognizer Agent"

    def identify_price_history_intent(self, prompt):
        """Identifies if the user is asking about price history, trends, or fluctuations."""
        record_interaction(self.name, f"Analyzing user prompt (English): '{prompt}' to identify the intent.")
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
            record_interaction(self.name, f"Identified intent: '{intent}'.")
            return intent
        except openai.OpenAIError as e:
            error_message = f"Error recognizing intent: {e}"
            record_interaction(self.name, error_message)
            st.error(error_message)
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

    # Clear previous turn's agent interactions
    st.session_state["agent_interactions"] = []

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Check if the prompt is likely in Spanish
        if any(char in "áéíóúñ¿¡" for char in prompt):
            record_interaction("Main Flow", "Detected potential Spanish input. Engaging Translator Agent.")
            english_prompt = translator_agent.translate_to_english(prompt)
            st.info(f"Translated from Spanish: '{prompt}' to English: '{english_prompt}'")
        else:
            english_prompt = prompt

        record_interaction("Main Flow", "Engaging Coin Identifier Agent to identify cryptocurrency.")
        identified_coin = coin_identifier_agent.identify_coin(english_prompt)

        record_interaction("Main Flow", "Engaging Intent Recognizer Agent to determine the user's intent.")
        intent = intent_recognizer_agent.identify_price_history_intent(english_prompt)

        if identified_coin and "price" in english_prompt.lower(): # Prioritize simple price requests
            record_interaction("Main Flow", f"Identified '{identified_coin}' and the user mentioned 'price'. Fetching latest price.")
            df = get_price_history(identified_coin)
            if df.empty:
                full_response = f"Sorry, I couldn't retrieve the price data for {identified_coin.capitalize()} right now."
                record_interaction("Main Flow", f"Failed to retrieve price data for '{identified_coin}'.")
            else:
                latest_price = df.iloc[-1]
                full_response = f"The latest price of {identified_coin.capitalize()} on {latest_price['Date']} is ${latest_price['Price (USD)']:.2f}."
                st.dataframe(df)
                record_interaction("Main Flow", f"Retrieved and displayed the latest price for '{identified_coin}'.")
        elif identified_coin and intent == "history":
            record_interaction("Main Flow", f"Identified '{identified_coin}' and the intent is to get price history. Engaging Crypto Information Agent.")
            if not crypto_info_agent.display_price_history(identified_coin):
                full_response = f"Sorry, I couldn't retrieve the price history for {identified_coin.capitalize()} right now."
                record_interaction("Main Flow", f"Failed to retrieve price history for '{identified_coin}'.")
        else:
            record_interaction("Main Flow", "No specific coin identified or the intent is general. Engaging Crypto Information Agent for a general response.")
            openai_response = crypto_info_agent.get_general_response(english_prompt)
            full_response = openai_response

        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Display agent interactions for the current turn after the assistant's response
    display_agent_interactions()