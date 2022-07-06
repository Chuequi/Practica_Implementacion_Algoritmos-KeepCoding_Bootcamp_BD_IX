import requests
import validators
import streamlit as st
import pandas as pd


class Session:
    pass


@st.cache(allow_output_mutation=True)
def fetch_session():
    session = Session()
    session.url = None
    session.predictions = None
    return session


session_state = fetch_session()


def validate_url(server_url):
    info_placeholder = st.empty()

    if server_url == "":
        # wait
        info_placeholder.write("")
    elif validators.url(server_url):
        # Save url
        session_state.url = server_url
        info_placeholder.write(
            "Perfect! Now we can start predicting. Insert your desired text below to make predictions:"
        )
    else:
        info_placeholder.write("Introduced text is not a URL, stopping...")
        st.stop()


def predict():
    input_text = st.text_area("Enter text")
    if input_text != "":
        payload = {"text": input_text}
        try:
            response = requests.post(session_state.url, json=payload)
            prediction = response.json()
            prediction["text"] = input_text
            if session_state.predictions is None:
                session_state.predictions = pd.DataFrame.from_dict(
                    {k: [v] for k, v in prediction.items()}
                )
            else:
                session_state.predictions = session_state.predictions.append(
                    {k: v for k, v in prediction.items()}, ignore_index=True
                )
            st.table(session_state.predictions)
            st.balloons()
        except Exception as ex:
            st.error(repr(ex))
        st.stop()


st.title("Sentiment Analysis Predictions")

st.markdown(
    "Welcome! With this app you can predict the sentiment of a given text using Deep Learning :smile:"
)

if session_state.url is None:
    st.write("Fist, paste below the predictor server URL: ")
else:
    st.write("Using cached server URL, change if desired:")

server_url = st.text_input(
    "Server URL", value=(session_state.url if session_state.url is not None else "")
)
session_state.url = server_url

if session_state.url is not None and server_url != "":
    validate_url(server_url)
    predict()
