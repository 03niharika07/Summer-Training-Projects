import streamlit as st
import joblib
import pandas as pd


# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="📖",
    layout="wide"
)


model = joblib.load("Model.pkl")


# ---------------- STYLE ----------------

st.markdown("""
<style>

.main{
    background:#faf8f2;
}


/* Header */

.header{
    background:#2f4858;
    padding:25px;
    border-radius:20px;
    color:white;
}


.header h1{
    margin-bottom:5px;
}

/* Result */

.score-box{
    background:#f4d35e;
    padding:30px;
    border-radius:25px;
    text-align:center;
}


.score{
    font-size:45px;
    font-weight:bold;
    color:#2f4858;
}


.label{
    font-size:18px;
    color:#555;
}


/* Button */

.stButton button{
    background:#2f4858;
    color:white;
    border-radius:12px;
    height:45px;
    width:100%;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)



# ---------------- HEADER ----------------

st.markdown("""

<h1>📖 Student Performance</h1>

<p>
Predict exam scores based on learning habits,
consistency and previous academic performance.
</p>

""", unsafe_allow_html=True)


st.write("")



# ---------------- LAYOUT ----------------

left,right = st.columns([1.2,1])



with left:

    st.markdown(
        "<div class='box'>",
        unsafe_allow_html=True
    )

    st.subheader("📝 Student Details")


    study_hours = st.number_input(
        "Study Hours / Week",
        0,100,20
    )


    attendance = st.number_input(
        "Attendance (%)",
        0,100,80
    )


    sleep_hours = st.number_input(
        "Sleep Hours / Day",
        0,24,7
    )


    internet_usage = st.number_input(
        "Internet Usage (Hours)",
        0,24,5
    )


    assignments_completed = st.number_input(
        "Assignments Completed",
        0,50,10
    )


    previous_score = st.number_input(
        "Previous Score",
        0,100,70
    )


    predict = st.button("Analyze Performance 🔍")


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )




with right:


    st.markdown(
        "<h2>📊 Prediction Result</h2>",
        unsafe_allow_html=True
    )


    if predict:


        data = pd.DataFrame({

            "study_hours":[study_hours],
            "attendance":[attendance],
            "sleep_hours":[sleep_hours],
            "internet_usage":[internet_usage],
            "assignments_completed":[assignments_completed],
            "previous_score":[previous_score]

        })


        result = model.predict(data)[0]

        result = round(result,2)


        st.markdown(f"""

        <div class="score-box">

        <div class="label">
        Predicted Exam Score
        </div>

        <div class="score">
        {result}/100
        </div>

        </div>

        """,
        unsafe_allow_html=True)



        st.write("")


        if result >=85:
            st.success("🏆 Excellent Academic Performance")

        elif result >=60:
            st.info("📈 Good Performance")

        else:
            st.warning("⚠ Improvement Required")



    else:

        st.info(
            "Enter student details and click Analyze Performance"
        )



st.write("")