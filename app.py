import streamlit as st
import openai

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

