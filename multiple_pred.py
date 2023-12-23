import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#loading the saved models
diabetes_model=pickle.load(open('C:/Users/User/Desktop/multiple diesease prediction system/saved models/diabetes_model.sav','rb'))
heart_disease_model=pickle.load(open('C:/Users/User/Desktop/multiple diesease prediction system/saved models/heart_disease_model.sav','rb'))
parkinson_disease_model=pickle.load(open('C:/Users/User/Desktop/multiple diesease prediction system/saved models/parkinsons_model.sav','rb'))
breast_cancer_model=pickle.load(open('C:/Users/User/Desktop/multiple diesease prediction system/saved models/breast_cancer_model.sav','rb'))


#sidebar for navigate
with st.sidebar:

    selected=option_menu('Multiple Disease Prediction System',['Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction','Breast_Cancer Classification'],icons=['activity','heart','person','arrow-right-circle-fill'],default_index=0)

#diabetes Preiction page
if(selected=='Diabetes Prediction'):
    st.title('Diabetes Prediction Using ML')

    col1,col2,col3=st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        Age = st.text_input('Age Of The Person')


    #code for prediction
    diab_diagnosis=''

    if st.button('Diabetes Test Result'):
        diab_prediction=diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

        if diab_prediction[0]==1:
            diab_diagnosis='The Person is Diabetic'
        else:
            diab_diagnosis='The Person is Not Diabetic'

    st.success(diab_diagnosis)

if(selected=='Heart Disease Prediction'):
    st.title('Heart Disease Prediction Using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age')

    with col2:
        sex = st.number_input('Sex')

    with col3:
        cp = st.number_input('Chest Pain types')

    with col1:
        trestbps = st.number_input('Resting Blood Pressure')

    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.number_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')

    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if (heart_prediction[0] == 1):
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)


if(selected=='Parkinsons Prediction'):
    st.title('Parkinsons Prediction Using ML')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinson_disease_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                           Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE,
                                                           DFA, spread1, spread2, D2, PPE]])

        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

if(selected=='Breast_Cancer Classification'):
    st.title('Breast_Cancer Classification Using ML')


    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        radius_mean = st.number_input('radius mean')

    with col2:
        texture_mean = st.number_input('texture_mean')

    with col3:
        perimeter_mean = st.number_input('perimeter_mean')

    with col4:
        area_mean= st.number_input('area_mean')

    with col5:
        smoothness_mean= st.number_input('smoothness_mean')

    with col1:
        compactness_mean = st.number_input('compactness_mean')

    with col2:
        concavity_mean = st.number_input('concavity_mean')

    with col3:
        concavepoints_mean = st.number_input('concave points_mean')

    with col4:
        symmetry_mean = st.number_input('symmetry_mean')

    with col5:
        fractal_dimension_mean = st.number_input('fractal_dimension_mean')

    with col1:
        radius_se = st.number_input('radius_se')

    with col2:
        texture_se = st.number_input('texture_se')

    with col3:
        perimeter_se = st.number_input('perimeter_se')

    with col4:
        area_se = st.number_input('area_se')

    with col5:
        smoothness_se = st.number_input('smoothness_se')

    with col1:
        compactness_se = st.number_input('compactness_se')

    with col2:
        concavity_se = st.number_input('concavity_se')

    with col3:

        concavepoints_se = st.number_input('concave points_se')

    with col4:
        symmetry_se = st.number_input('symmetry_se')

    with col5:

        fractal_dimension_se = st.number_input('fractal_dimension_se')

    with col1:

        radius_worst = st.number_input('radius_worst')

    with col2:

        texture_worst = st.number_input('texture_worst')

    with col3:
        perimeter_worst = st.number_input('perimeter_worst')

    with col4:
        area_worst = st.number_input('area_worst')

    with col5:
        smoothness_worst = st.number_input('smoothness_worst')

    with col1:
        compactness_worst = st.number_input('compactness_worst')

    with col2:
        concavity_worst = st.number_input('concavity_worst')
    with col3:
        concavepoints_worst = st.number_input('concave points_worst')
    with col4:
        symmetry_worst = st.number_input('symmetry_worst')
    with col5:
        fractal_dimension_worst = st.number_input('fractal_dimension_worst')

    # code for Prediction
    cancer_diagnosis = ''

    # creating a button for Prediction

    if st.button('Breast Cancer Test Result'):
        cancer_prediction = breast_cancer_model.predict(
            [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concavepoints_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concavepoints_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concavepoints_worst, symmetry_worst, fractal_dimension_worst]])

        if (cancer_prediction[0] == 1):
            cancer_diagnosis = 'The Breast Cancer is Benign'
        else:
            cancer_diagnosis = 'The Breast Cancer is Malignant'

    st.success(cancer_diagnosis)


