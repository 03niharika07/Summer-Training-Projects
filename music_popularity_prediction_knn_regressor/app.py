import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="Music Popularity Prediction",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Music Popularity Prediction")
st.write("Predict Spotify Song Popularity using KNN Regressor")

# USER INPUTS

duration_ms = st.number_input(
    "Duration (ms)",
    min_value=0,
    value=200000
)

explicit_input = st.selectbox(
    "Explicit Content",
    ["False", "True"]
)

danceability = st.slider(
    "Danceability",
    0.0, 1.0, 0.50
)

energy = st.slider(
    "Energy",
    0.0, 1.0, 0.50
)

key = st.selectbox(
    "Key",
    list(range(12))
)

loudness = st.number_input(
    "Loudness",
    value=-10.0
)

mode = st.selectbox(
    "Mode",
    [0, 1]
)

speechiness = st.slider(
    "Speechiness",
    0.0, 1.0, 0.10
)

acousticness = st.slider(
    "Acousticness",
    0.0, 1.0, 0.50
)

instrumentalness = st.slider(
    "Instrumentalness",
    0.0, 1.0, 0.00
)

liveness = st.slider(
    "Liveness",
    0.0, 1.0, 0.20
)

valence = st.slider(
    "Valence",
    0.0, 1.0, 0.50
)

tempo = st.number_input(
    "Tempo",
    value=120.0
)

time_signature = st.selectbox(
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

genre = st.selectbox(
    "Track Genre",
    genre_list
)

# PREDICTION

if st.button("Predict Popularity"):

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
        f"🎯 Predicted Popularity Score: {prediction:.2f}/100"
    )