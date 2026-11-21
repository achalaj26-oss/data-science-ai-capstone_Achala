
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from config import MODEL_METRICS, VALIDATION_FILE, MODEL_FILE, IMPUTER_FILE

st.set_page_config(
    page_title="AMEX Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

selected_features = [
    "P_2_last", "D_48_last", "B_2_last", "D_61_last", "B_9_last",
    "B_18_last", "D_55_last", "D_61_mean", "B_2_mean", "D_44_last",
    "B_9_mean", "R_1_mean", "B_3_last", "D_75_last", "B_7_last",
    "R_1_last", "B_4_last", "B_20_last", "B_19_last", "B_38_last",
    "B_1_last", "B_23_mean", "R_2_last", "B_22_mean", "R_10_mean",
    "B_17_mean", "B_38_mean", "B_22_last", "B_1_mean", "B_30_last"
]

@st.cache_data
def load_data():
    return pd.read_csv(VALIDATION_FILE)

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model

@st.cache_resource
def load_medians():
    imputer = joblib.load(IMPUTER_FILE)
    return imputer.statistics_

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

validation_data = load_data()
xgb_model = load_model()
training_medians = load_medians()
shap_explainer = load_explainer(xgb_model)

st.title("💳 Credit Card Default Risk Prediction")
st.subheader("Explainable Machine Learning for Credit Risk Assessment")

st.markdown(
    "An interactive application for credit default risk prediction "
    "using an XGBoost machine learning model."
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    ["Dashboard Overview", "Customer Prediction", "Model Explainability"]
)

if page == "Dashboard Overview":

    st.header("Model Performance Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{MODEL_METRICS['Accuracy']:.2%}")
    col2.metric("Precision", f"{MODEL_METRICS['Precision']:.2%}")
    col3.metric("Recall", f"{MODEL_METRICS['Recall']:.2%}")
    col4.metric("F1 Score", f"{MODEL_METRICS['F1 Score']:.2%}")
    col5.metric("ROC-AUC", f"{MODEL_METRICS['ROC-AUC']:.2%}")

    st.divider()

    st.header("Validation Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Validation Customers", f"{len(validation_data):,}")
    col2.metric("Features Used", "30")
    col3.metric("Model", "XGBoost")

    st.info(
        "The model was evaluated using a validation dataset of "
        "15,000 customers. The metrics displayed above represent "
        "the model's evaluation results."
    )

    st.divider()

    st.divider()

    st.header("Customer Class Distribution")

    class_counts = validation_data["target"].value_counts().sort_index()

    class_distribution = pd.DataFrame({
        "Customer Class": ["Non-Default", "Default"],
        "Number of Customers": [
            class_counts.get(0, 0),
            class_counts.get(1, 0)
        ]
    })

    fig_dist = px.bar(
        class_distribution,
        x="Customer Class",
        y="Number of Customers",
        text="Number of Customers",
        title="Customer Distribution",
        height=350
    )

    fig_dist.update_traces(
        texttemplate="%{text:,}",
        textposition="outside"
    )

    fig_dist.update_layout(
        template="plotly_dark",
        title_font_size=18,
        font_size=13,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )

    st.plotly_chart(
        fig_dist,
        use_container_width=True
    )

    st.caption(
        "Distribution of default and non-default customers "
        "in the validation dataset."
    )

    st.header("Confusion Matrix")
    

    confusion_matrix_data = np.array([
        [10263, 853],
        [839, 3045]
    ])

    fig, ax = plt.subplots(figsize=(5, 3))

    sns.heatmap(
    confusion_matrix_data,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Default", "Default"],
    yticklabels=["Non-Default", "Default"],
    ax=ax
   )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Validation Confusion Matrix")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    

    

elif page == "Customer Prediction":

    st.header("Individual Customer Prediction")

    customer_id = st.selectbox(
        "Select Customer ID",
        validation_data["customer_ID"].tolist()
    )

    if st.button("Predict Default Risk", type="primary"):

        customer = validation_data[
            validation_data["customer_ID"] == customer_id
        ]

        customer_features = customer[selected_features]

        customer_features_imputed = customer_features.fillna(
            pd.Series(training_medians, index=selected_features)
        )

        probabilities = xgb_model.predict_proba(
            customer_features_imputed
        )

        default_probability = probabilities[0][1]
        non_default_probability = probabilities[0][0]

        predicted_class = (
            "DEFAULT" if default_probability >= 0.5
            else "NON-DEFAULT"
        )

        st.divider()
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Default Probability", f"{default_probability:.2%}")
        col2.metric("Non-default Probability", f"{non_default_probability:.2%}")
        col3.metric("Predicted Classification", predicted_class)

        st.progress(float(default_probability))

        if predicted_class == "DEFAULT":
            st.warning(
                "The model classifies this customer as having "
                "a higher predicted default risk at the 0.5 threshold."
            )
        else:
            st.success(
                "The model classifies this customer as non-default "
                "at the 0.5 threshold."
            )

        st.caption(
            "This prediction is a model estimate and should not be "
            "considered a guarantee of future customer behaviour."
        )

elif page == "Model Explainability":

    st.header("Customer Prediction Explanation")

    st.subheader("Global Feature Importance")

    global_shap_file = os.path.join(
        os.path.dirname(VALIDATION_FILE),
        "global_shap_importance.csv"
    )

    global_shap_df = pd.read_csv(global_shap_file)

    top_global_features = global_shap_df.head(10).sort_values(
        "Mean_Absolute_SHAP",
        ascending=True
    )

    fig_global, ax_global = plt.subplots(figsize=(5, 3.5))

    ax_global.barh(
    top_global_features["Feature"],
    top_global_features["Mean_Absolute_SHAP"],
    height=0.6
    ) 

    ax_global.set_xlabel("Mean Absolute SHAP Value", fontsize=9)
    ax_global.set_ylabel("Feature", fontsize=9)
    ax_global.set_title("Top 10 Global Feature Importance", fontsize=11)

    ax_global.tick_params(axis="both", labelsize=9)
    plt.tight_layout()

    st.pyplot(fig_global, width=800)
    plt.close(fig_global)

    st.caption(
        "This chart shows the average magnitude of each feature's "
        "contribution to model predictions across the sampled customers. "
        "It does not indicate causality."
    )

    st.divider()

    customer_id = st.selectbox(
        "Select Customer",
        validation_data["customer_ID"].tolist(),
        key="shap_customer"
    )

    if st.button("Generate Explanation", type="primary"):

        customer = validation_data[
            validation_data["customer_ID"] == customer_id
        ]

        customer_features = customer[selected_features]

        customer_features_imputed = customer_features.fillna(
            pd.Series(training_medians, index=selected_features)
        )

        probabilities = xgb_model.predict_proba(
            customer_features_imputed
        )

        default_probability = float(probabilities[0][1])
        non_default_probability = float(probabilities[0][0])

        predicted_class = (
            "DEFAULT" if default_probability >= 0.5
            else "NON-DEFAULT"
        )

        explanation = shap_explainer(
            customer_features_imputed
        )

        shap_values = explanation.values[0]
        feature_values = customer_features_imputed.iloc[0].values

        # Create explanation dataframe
        explanation_df = pd.DataFrame({
            "Feature": selected_features,
            "Feature Value": feature_values,
            "SHAP Contribution": shap_values
        })

        explanation_df["Direction"] = np.where(
            explanation_df["SHAP Contribution"] > 0,
            "Towards Default",
            "Away from Default"
        )

        explanation_df["Absolute Contribution"] = (
            explanation_df["SHAP Contribution"].abs()
        )

        explanation_df = explanation_df.sort_values(
            "Absolute Contribution",
            ascending=False
        )

        # Prediction summary
        st.subheader("Prediction Summary")

        col1, col2 = st.columns(2)

        col1.metric(
            "Estimated Default Probability",
            f"{default_probability:.2%}"
        )

        col2.metric(
            "Predicted Classification",
            predicted_class
        )

        if predicted_class == "DEFAULT":
            st.warning(
                "The model predicts default at the 0.5 classification threshold."
            )
        else:
            st.success(
                "The model predicts non-default at the 0.5 classification threshold."
            )

        st.progress(default_probability)

        st.caption(
            "The probability is a model estimate, not a guarantee "
            "of the customer's future behaviour."
        )

        st.divider()

        # Top 5 feature explanations
        st.subheader("Top 5 Factors Influencing the Prediction")

        top5 = explanation_df.head(5).copy()

        top5["SHAP Contribution"] = top5[
            "SHAP Contribution"
        ].round(4)

        top5["Feature Value"] = top5[
            "Feature Value"
        ].round(4)

        display_df = top5[
            ["Feature", "Feature Value", "SHAP Contribution", "Direction"]
        ]

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )

        # Top 5 chart
        st.subheader("Feature Contribution Visualisation")

        chart_data = top5.sort_values(
            "SHAP Contribution",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(4, 3))

        ax.barh(
        chart_data["Feature"],
        chart_data["SHAP Contribution"],
        height=0.6
      )

        ax.axvline(0, linestyle="--", linewidth=0.8)

        ax.set_xlabel("SHAP Contribution (Model Output Scale)", fontsize=8)
        ax.set_ylabel("Feature", fontsize=8)
        ax.set_title("Top 5 Feature Contributions", fontsize=9)
        # Reduce tick label sizes
        ax.tick_params(axis="both", labelsize=8)
        plt.tight_layout()

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.caption(
            "Positive SHAP contributions push the model output towards "
            "the default class, while negative contributions push it away. "
            "SHAP values represent model contributions and are not percentages."
        )

        # Detailed explanation
        with st.expander("View Detailed SHAP Waterfall Explanation"):

            st.markdown(
                "This plot provides a more detailed explanation of how "
                "individual feature contributions combine to produce "
                "the model output."
            )

            shap.plots.waterfall(
                explanation[0],
                max_display=10,
                show=False
            )

            st.pyplot(plt.gcf(),width=800)
            plt.close()

        st.caption(
            "SHAP explains the model's behaviour. It does not establish "
            "causal relationships between features and default."
        )
