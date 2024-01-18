import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import re
import pickle
import time
from streamlit_extras.let_it_rain import rain

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon=":anatomical_heart:",
)
progress_text = "Loading..."
my_bar = st.progress(0, text=progress_text)
for percent_complete in range(100):
    time.sleep(0.05)
    my_bar.progress(percent_complete + 1, text=f"{percent_complete + 1}%  Loading...")
my_bar.empty()

st.write(
    """
    # Heart Disease Prediction System :anatomical_heart:
    This web app is capable of forecasting the probability of getting a heart disease and it is trained using 6 different type of models listed here: Neural Network (Default Model), Logistic Regression, Decision Tree, Random Forest, XG Boost and K Nearest Neighbors.
    Neural Network is the default model used to train the dataset as it has the highest accuracy among the other models. Nevertheless,
    you could choose any of the models listed above to train the model by clicking the dropdown menu on the sidebar :point_left: .
    Get to know the models accuracy score with the [data visualization here](#models-accuracy-score).
    """
)

st.write(
    """
    ##### About the Dataset
    The data used to train the model were obtained from Kaggle. You can check out the dataset [here](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset).
    If you are interested in the code, you can check the jupyter notebook [here](https://github.com/Melo04/heart-disease-prediction/blob/main/notebook.ipynb).
    """
)

class Toc:
    def __init__(self):
        self._items = []
        self._placeholder = None

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
    
    def _markdown(self, text, level, space=""):
        key = re.sub('[^0-9a-zA-Z]+', '-', text).lower()

        style = 'font-size:1.5rem; font-weight:600; color:rgb(139, 255, 114); line-height 1.2;"'
        st.markdown(f"<p id='{key}' style={style}>{text}</p>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")

# place table of contents at sidebar
toc = Toc()
st.sidebar.title("Table of contents")
toc.placeholder(sidebar=True)
choose_model = st.sidebar.selectbox("Choose a model to train", ["Neural Network (Default)", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "K Nearest Neighbors"])

st.markdown('---')
toc.header('Data Dictionary 📖')
st.write("""
        | Variable | Definition | Key |
        | --- | --- | --- |
        | Age | Age Of the patient | |
        | Sex | Sex of the patient | 0 = male; 1 = female; |
        | exang | exercise induced anigna | 0 = no; 1 = yes; |
        | caa | number of major vessels | 0-3 |
        | cp | Chest Pain type | 1 = typical anigma; 2 = atypical anigma; 3 = non-anginal pain; 4 = asymptomatic |
        | trtbps | resting blood pressure (in mm Hg) |  |
        | chol | cholestoral in mg/dl fetched via BMI sensor |  |
        | fbs | fasting blood sugar > 120 mg/dl | 1 = true; 0 = false |
        | rest_ecg | resting electrocardiographic results | 0 = normal; 1 = having ST-T wave abnormality; 2=showing probable or definite left ventricular hypertropy |
        | thalachh | maximum heart rate achieved | |
        | exng | exercise induced angina | 1 = yes; 0 = no |
        | oldpeak | previous peak | |
        | slp | slope | 0 = upsloping; 1 = flat; 2 = downsloping |
        | thall | Thal rate | 0 = normal; 1 = fixed defect; 2 = reversible defect |
        | target | chance of getting a heart attack | 0 = low chance; 1 = high chance |
        """)

data = pd.read_csv('data/processed-data.csv')

data_table = pd.DataFrame({
    'Age': data.age,
    'Sex': data.sex,
    'Cp': data.cp,
    'Trtbps': data.trtbps,
    'Chol': data.chol,
    'Fbs': data.fbs,
    'Restecg': data.restecg,
    'Thalachh': data.thalachh,
    'Exng': data.exng,
    'OldPeak': data.oldpeak,
    'Slp': data.slp,
    'Caa': data.caa,
    'Thall': data.thall,
})

@st.cache_data
def load_data(nrows):
    df = pd.read_csv('data/processed-data.csv', nrows=nrows)
    return df

df = load_data(10000)
st.markdown('---')
toc.header('Dataset 📌')
st.write(df)

st.markdown('---')
toc.header('Data Visualisation 📊')

toc.subheader('Models Accuracy Score')
fig, ax = plt.subplots()
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'K-Nearest Neighbor', 'Neural Network']
accuracy = [86.89, 83.61, 85.25, 86.89, 91.80, 92.116]
bar_colors=['#A9BCD0', '#1C2541', '#3A506B', '#5BC0BE', '#6FFFE9']
bars = ax.bar(models, accuracy, label=models, color=bar_colors, width=0.6)
plt.xticks(rotation=45, ha='right')
ax.set_ylabel('Accuracy')
ax.set_title('Models Accuracy Score')
ax.legend(title='Models')
for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{acc:.2f}%', ha='center', va='bottom')
st.pyplot(fig)
st.markdown('---')

toc.subheader('Heart Disease Percentage')
f, ax = plt.subplots(1, 2, figsize=(16,8))
df["output"].replace({0:"No Heart Disease", 1:"Heart Disease"}).value_counts().plot(kind='pie', colors=["#ACFF94", "#80FFE9"], ax=ax[0], explode=[0,0.1], autopct='%1.2f%%', shadow=True)
df["output"].replace({0:"No Heart Disease", 1:"Heart Disease"}).value_counts().plot(kind='bar', color=["#ACFF94", "#80FFE9"], ax=ax[1])
ax[0].set_ylabel('')
ax[1].set_ylabel('Count')
ax[1].set_xlabel('')
ax[0].set_title('Heart Disease Percentage')
ax[1].set_title('Heart Disease Count')
st.pyplot(f)
st.markdown('---')

toc.subheader('Correlation Matrix of the Dataset')
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
plt.yticks(rotation=0)
st.pyplot(fig)
st.markdown('---')

st.write("""Density Distribution of Age""")
plt.figure(figsize=(10, 6))
fg = sns.displot(df.age, color="#8700FC", label="Age", kde=True)
plt.legend()
st.pyplot(fg)


st.markdown('---')
toc.header('Predict Likelihood of getting a Heart Disease 🤞')
st.write("You can compare the models by taking note of the probability for getting a heart disease. Take note that decision tree and random forest has the lowest accuracy, hence their prediction might not be accurate.")
st.write("""Sample dataset from Kaggle: """)
st.code("High Possibility Of Getting a Heart Disease: 63, female, asymptomatic, 145, 233, true, normal, 150, no, 2.3, downsloping, 0, 1   ")
st.code("Low Possibility Of Getting a Heart Disease: 43, male, typical angina, 132, 341, true, normal, 136, yes, 3.0, flat, 0, 3   ")
form = st.form(key='my_form')

toc.generate()

def user_input_features():
    age = form.slider('Age', 1, 100, 50)
    sex =  form.selectbox('Sex', ('Male', 'Female'))
    cp =  form.selectbox('Chest Pain Type', ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'))
    trtbps =  form.slider('Resting Blood Pressure', 94, 200, 130)
    chol =  form.slider('Cholestoral', 126, 564, 246)
    fbs = form.selectbox('Fasting Blood Sugar', ('True', 'False'))
    restecg =  form.selectbox('Resting Electrocardiographic Results', ('Normal', 'Having ST-T Wave Abnormality', 'Left Ventricular Hypertropy'))
    thalachh =  form.slider('Maximum Heart Rate Achieved', 71, 202, 149)
    exng =  form.selectbox('Exercise Induced Angina', ('Yes', 'No'))
    oldpeak = form.slider('Old Peak', 0.0, 6.2, 1.0)
    slp =form.selectbox('Slope', ('Upsloping', 'Flat', 'Downsloping'))
    caa = form.slider('Number of Major Vessels', 0, 3, 1)
    thall = form.slider('Thal Rate (Thallium Stress Test)', 0, 3, 1)

    sex_num = 1 if sex == 'Male' else 0
    cp_num = 0 if cp == 'Typical Angina' else 1 if cp == 'Atypical Angina' else 2 if cp == 'Non-Anginal Pain' else 3
    fbs_num = 1 if fbs == 'True' else 0
    restecg_num = 0 if restecg == 'Normal' else 1 if restecg == 'Having ST-T Wave Abnormality' else 2
    exng_num = 1 if exng == 'Yes' else 0
    slp_num = 0 if slp == 'Upsloping' else 1 if slp == 'Flat' else 2

    input_data = {
        'age': age,
        'sex': sex_num,
        'cp': cp_num,
        'trtbps': trtbps,
        'chol': chol,
        'fbs': fbs_num,
        'restecg': restecg_num,
        'thalachh': thalachh,
        'exng': exng_num,
        'oldpeak': oldpeak,
        'slp': slp_num,
        'caa': caa,
        'thall': thall,
    }
    return input_data

df_input = user_input_features()
predict = form.form_submit_button(label='Predict :crossed_fingers:')

def open_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

if predict:
    user_input = pd.DataFrame(data=df_input, index=[1])
    if choose_model == "Neural Network (Default)":
        filename = './model/neural_network_model.h5'
        model = tf.keras.models.load_model(filename)
    elif choose_model == "Logistic Regression":
        filename = './model/logistic_model.pkl'
    elif choose_model == "Decision Tree":
        filename = './model/decision_tree_model.pkl'
    elif choose_model == "Random Forest":
        filename = './model/random_forest_model.pkl'
    elif choose_model == "XGBoost":
        filename = './model/xgb_model.pkl'
    elif choose_model == "K Nearest Neighbors":
        filename = './model/knn_model.pkl'
    if choose_model != "Neural Network (Default)":
        model = open_model(filename)

    prediction = model.predict(user_input.values)
    st.write("Using model: ")
    if choose_model == "XGBoost":
        st.write("XGBoost")
    else:
        st.write(model)
    if choose_model == "Neural Network (Default)":
        threshold = 0.5
        binary_predictions = (prediction > threshold).astype(int)
        attack = prediction[0][0]
        if binary_predictions[0] == 0:
            rain(
                emoji="🎉",
                font_size=54,
                falling_speed=5,
                animation_length=0.5,
            )
            st.success(f'Congrats, You have a low probability of getting a heart disease with a probability of {100 - (attack*100):.2f}% :tada:')
        else:
            rain(
                emoji="💀",
                font_size=54,
                falling_speed=5,
                animation_length=0.5,
            )
            st.error(f'RIP, You have a high possibility of getting a heart disease with a probability of {(attack*100):.2f}% :skull:')
    else:
        probability = model.predict_proba(user_input.values)
        if prediction[0] == 0:
            rain(
                emoji="🎉",
                font_size=54,
                falling_speed=5,
                animation_length=0.5,
            )
            st.success(f'Congrats, You have a low possibility of getting a heart disease with a probability of {(probability[0][0]*100):.2f}% :tada:')
        else:
            rain(
                emoji="💀",
                font_size=54,
                falling_speed=5,
                animation_length=0.5,
            )
            st.error(f'RIP, You have a high possibility of getting a heart disease with a probability of {(probability[0][1]*100):.2f}% :skull:')