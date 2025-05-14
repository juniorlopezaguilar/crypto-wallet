import streamlit as st
import openai
import requests
import pandas as pd
from openai import OpenAI
from datetime import datetime, timedelta

st.header("Demo Crypto chat with multiple agents")
st.write("This chat works with multiple chat ai agents")

# ------------ Add a Reload/Reset button in the sidebar
with st.sidebar:
    st.subheader("App Controls")
    if st.button("Reload/Reset App"):
        st.session_state.pop("messages", None) # Clear chat history
        st.session_state.pop("agent_interactions", None) # Clear agent interactions
        st.cache_data.clear() # Clear cached data (including price history)
        st.rerun() # Rerun the app from the top

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
        st.subheader("Agent Interactions (Current Turn)") # Updated title
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
            "sol": "solana", # Added solana mapping
            "solana": "solana",
            "ada": "cardano", # Added cardano mapping
            "cardano": "cardano",
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
                    {"role": "system", "content": "You are a helpful assistant that identifies if the user is asking about a specific digital coin. If so, return only the lowercase CoinGecko ID of the coin. If not, or if you are unsure, return 'none'."}, # Made instructions more specific
                    {"role": "user", "content": f"Is the user asking about a specific digital coin in this prompt: '{prompt}'? If yes, which one (return its CoinGecko ID)?"}
                ],
                max_tokens=30,
                temperature=0.3
            )
            identification = response.choices[0].message.content.strip().lower()
            record_interaction(self.name, f"OpenAI responded with: '{identification}'.")
            if identification and identification != "none":
                 # Check direct mapping first, then potentially use the identified string if not in mapping
                coingecko_id = self.coin_mapping.get(identification, identification) # Use identified string if not in mapping
                # Simple validation: check if it looks like a possible ID (e.g., not a full sentence)
                if ' ' not in coingecko_id and len(coingecko_id) > 1:
                     record_interaction(self.name, f"Identified cryptocurrency: '{identification}', using CoinGecko ID: '{coingecko_id}'.")
                     return coingecko_id
                else:
                    record_interaction(self.name, f"Identified '{identification}' but it doesn't look like a valid CoinGecko ID.")
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
                messages=[
                     {"role": "system", "content": "You are a helpful assistant knowledgeable about cryptocurrency wallets and digital coins. Provide concise answers."}, # Added system message
                     {"role": "user", "content": prompt}
                     ],
                max_tokens=200, # Increased max tokens slightly for better general answers
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
            # Add a small summary if possible (optional)
            try:
                start_price = df.iloc[0]['Price (USD)']
                end_price = df.iloc[-1]['Price (USD)']
                change = end_price - start_price
                percentage_change = (change / start_price) * 100 if start_price != 0 else 0
                st.markdown(f"Over the last 7 days ({df.iloc[0]['Date']} to {df.iloc[-1]['Date']}), the price changed by ${change:.2f} ({percentage_change:.2f}%).")
            except Exception as e:
                record_interaction(self.name, f"Could not generate price summary: {e}")
                pass # Ignore summary errors
            return True
        else:
            warning_message = f"Could not retrieve price history for {coin_id.capitalize()}. Please check the coin ID."
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
                    {"role": "system", "content": "You are a helpful assistant that translates the user's input accurately from Spanish to English. Provide only the translated text."}, # Made instructions more specific
                    {"role": "user", "content": f"Translate the following Spanish text to English: '{text}'"}
                ],
                max_tokens=100, # Increased tokens slightly
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
                    {"role": "system", "content": "You are a helpful assistant that determines if the user is asking specifically about the price history, past trends, or historical fluctuations of a cryptocurrency. If the user is asking for *current* price, return 'general'. Return 'history' only if they are asking about past price data. Return 'general' otherwise."}, # Improved instructions
                    {"role": "user", "content": f"Determine the intent of this prompt: '{prompt}'. Is it 'history' or 'general'?"}
                ],
                max_tokens=20,
                temperature=0.1 # Even lower temperature for stricter classification
            )
            intent = response.choices[0].message.content.strip().lower()
            # Simple validation to ensure it's one of the expected intents
            if intent not in ['history', 'general']:
                 intent = 'general' # Default to general if unexpected response
                 record_interaction(self.name, f"Unexpected intent response '{intent}', defaulting to 'general'.")
            else:
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
        # A simple check for common Spanish characters and words
        spanish_indicators = ["¿", "¡", "á", "é", "í", "ó", "ú", "ñ", "que", "de", "la", "el", "un", "una"]
        is_spanish = any(indicator in prompt.lower() for indicator in spanish_indicators) or any(char.islower() and char in "áéíóúñ" for char in prompt)


        if is_spanish:
            record_interaction("Main Flow", "Detected potential Spanish input. Engaging Translator Agent.")
            english_prompt = translator_agent.translate_to_english(prompt)
            # Only show the translation if it's different from the original prompt
            if english_prompt.strip().lower() != prompt.strip().lower():
                st.info(f"Translated from Spanish: '{prompt}' to English: '{english_prompt}'")
            else:
                 record_interaction("Main Flow", "Translation resulted in text similar to the original. Proceeding with original.")
                 english_prompt = prompt # Use original if translation seems off
        else:
            english_prompt = prompt

        record_interaction("Main Flow", "Engaging Coin Identifier Agent to identify cryptocurrency.")
        identified_coin = coin_identifier_agent.identify_coin(english_prompt)

        # Only run intent recognizer if a coin was identified
        if identified_coin:
            record_interaction("Main Flow", "Engaging Intent Recognizer Agent to determine the user's intent.")
            intent = intent_recognizer_agent.identify_price_history_intent(english_prompt)

            if intent == "history":
                record_interaction("Main Flow", f"Identified '{identified_coin}' and the intent is to get price history. Engaging Crypto Information Agent.")
                if not crypto_info_agent.display_price_history(identified_coin):
                    full_response = f"Sorry, I couldn't retrieve the price history for {identified_coin.capitalize()} right now."
                    record_interaction("Main Flow", f"Failed to retrieve price history for '{identified_coin}'.")
                else:
                     full_response = f"Here is the price history for {identified_coin.capitalize()} over the last 7 days." # Provide a basic response after displaying data
                     record_interaction("Main Flow", f"Successfully retrieved and displayed price history for '{identified_coin}'.")
            else: # Intent is 'general' or fallback
                 record_interaction("Main Flow", f"Identified '{identified_coin}' but intent is not price history ('{intent}'). Engaging Crypto Information Agent for a general response about {identified_coin}.")
                 openai_response = crypto_info_agent.get_general_response(english_prompt) # Use English prompt for general response
                 full_response = openai_response
        else:
            record_interaction("Main Flow", "No specific coin identified. Engaging Crypto Information Agent for a general response.")
            openai_response = crypto_info_agent.get_general_response(english_prompt) # Use English prompt for general response
            full_response = openai_response


        # Fallback response if full_response is still empty (shouldn't happen with current logic but good practice)
        if not full_response:
             full_response = "I'm sorry, I couldn't process your request."
             record_interaction("Main Flow", "Fallback: Response is empty.")


        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Display agent interactions for the current turn after the assistant's response
    display_agent_interactions()