import streamlit as st
import pandas as pd
import joblib

# Load Files
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Page Config
st.set_page_config(
    page_title="Music Popularity Prediction",
    page_icon="🎵",
    layout="centered"
)

# Main Page
st.title("🎵 Music Popularity Prediction")
st.markdown(
    "Predict the popularity score of a Spotify track using **KNN Regressor**."
)

# Sidebar
st.sidebar.header("🎧 Song Features")

duration_ms = st.sidebar.number_input(
    "Duration (ms)",
    min_value=0,
    value=200000
)

explicit_input = st.sidebar.selectbox(
    "Explicit Content",
    ["False", "True"]
)

danceability = st.sidebar.slider(
    "Danceability",
    0.0, 1.0, 0.50
)

energy = st.sidebar.slider(
    "Energy",
    0.0, 1.0, 0.50
)

key = st.sidebar.selectbox(
    "Key",
    list(range(12))
)

loudness = st.sidebar.number_input(
    "Loudness",
    value=-10.0
)

mode = st.sidebar.selectbox(
    "Mode",
    [0, 1]
)

speechiness = st.sidebar.slider(
    "Speechiness",
    0.0, 1.0, 0.10
)

acousticness = st.sidebar.slider(
    "Acousticness",
    0.0, 1.0, 0.50
)

instrumentalness = st.sidebar.slider(
    "Instrumentalness",
    0.0, 1.0, 0.00
)

liveness = st.sidebar.slider(
    "Liveness",
    0.0, 1.0, 0.20
)

valence = st.sidebar.slider(
    "Valence",
    0.0, 1.0, 0.50
)

tempo = st.sidebar.number_input(
    "Tempo",
    value=120.0
)

time_signature = st.sidebar.selectbox(
    "Time Signature",
    [3, 4, 5]
)

genre_list = [
    "acoustic","afrobeat","alt-rock","alternative","ambient",
    "anime","black-metal","bluegrass","blues","brazil",
    "breakbeat","british","cantopop","chicago-house","children",
    "chill","classical","club","comedy","country",
    "dance","dancehall","death-metal","deep-house","detroit-techno",
    "disco","disney","drum-and-bass","dub","dubstep",
    "edm","electro","electronic","emo","folk",
    "forro","french","funk","garage","german",
    "gospel","goth","grindcore","groove","grunge",
    "guitar","happy","hard-rock","hardcore","hardstyle",
    "heavy-metal","hip-hop","honky-tonk","house","idm",
    "indian","indie","indie-pop","industrial","iranian",
    "j-dance","j-idol","j-pop","j-rock","jazz",
    "k-pop","kids","latin","latino","malay",
    "mandopop","metal","metalcore","minimal-techno","mpb",
    "new-age","opera","pagode","party","piano",
    "pop","pop-film","power-pop","progressive-house","psych-rock",
    "punk","punk-rock","r-n-b","reggae","reggaeton",
    "rock","rock-n-roll","rockabilly","romance","sad",
    "salsa","samba","sertanejo","show-tunes","singer-songwriter",
    "ska","sleep","songwriter","soul","spanish",
    "study","swedish","synth-pop","tango","techno",
    "trance","trip-hop","turkish","world-music"
]

genre = st.sidebar.selectbox(
    "Track Genre",
    genre_list
)

st.sidebar.markdown("---")

predict_btn = st.sidebar.button("🎯 Predict Popularity")

# Prediction
if predict_btn:

    explicit = label_encoder.transform([explicit_input])[0]

    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    input_df["duration_ms"] = duration_ms
    input_df["explicit"] = explicit
    input_df["danceability"] = danceability
    input_df["energy"] = energy
    input_df["key"] = key
    input_df["loudness"] = loudness
    input_df["mode"] = mode
    input_df["speechiness"] = speechiness
    input_df["acousticness"] = acousticness
    input_df["instrumentalness"] = instrumentalness
    input_df["liveness"] = liveness
    input_df["valence"] = valence
    input_df["tempo"] = tempo
    input_df["time_signature"] = time_signature

    genre_col = f"track_genre_{genre}"

    if genre_col in input_df.columns:
        input_df[genre_col] = 1

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.success(
        f"🎵 Predicted Popularity Score: {prediction:.2f}/100"
    )