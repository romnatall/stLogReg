
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from LogReg import LogReg
import streamlit as st


# Загрузка данных
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")



@st.cache_resource
def load_data():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        lr = LogReg(learning_rate=0.01, n_inputs=2)
        scaler = StandardScaler()
        X = scaler.fit_transform(df[['CCAvg', 'Income']])
        y = df['Personal.Loan'].to_numpy()
        lr.fit(X, df[['Personal.Loan']].to_numpy(), epochs=1000)
        

    else:
        # Используйте стандартный датасет, если файл не загружен
        df = pd.read_csv('aux/credit_train.csv')
        scaler = StandardScaler()
        X = scaler.fit_transform(df[['CCAvg', 'Income']])
        y=df[['Personal.Loan']].to_numpy()
        lr = LogReg(learning_rate=0.01, n_inputs=2)
        lr.fit(X, y, epochs=1000)
        #lr.intercept_=0.029916065767719158
        #lr.coef_=[1.3178887 , 2.36463115]
        

    # Визуализация данных
    st.subheader("Визуализация данных")
    fig=plt.figure(figsize=(10, 7))
    sns.scatterplot(x='CCAvg', y='Income', hue='Personal.Loan', data=df)
    plt.legend(title='Возврат кредита', labels=['есть', 'дефолт'])
    plt.xlabel('Кредитный рейтинг')
    plt.ylabel('Доход')

    st.pyplot(fig)

    # Масштабирование данных

    # Генерация точек для линии
    x_line_manual = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y_line_manual = (-lr.coef_[0] * x_line_manual - lr.intercept_) / lr.coef_[1]
    return df, lr, scaler,X


def otherstuff(df, lr, scaler ,X):
    # Предсказание вероятности дефолта
    st.subheader("Предсказание вероятности дефолта")
    ccavg_input = st.slider("Выберите значение кредитного рейтинга", float(df['CCAvg'].min()), float(df['CCAvg'].max()), float(df['CCAvg'].mean()))
    income_input = st.slider("Выберите значение дохода", float(df['Income'].min()), float(df['Income'].max()), float(df['Income'].mean()))

    scaled_input = scaler.transform([[ccavg_input, income_input]])
    probability_default = lr.predict(scaled_input)[0]
    st.write(f"Вероятность дефолта: {1-probability_default:.2%}")

    # Визуализация линии решения
    # Генерация точек для линии (для вашей модели)
    x_line_lr = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)

    # Генерация точек для линии (для вашей ручной реализации)
    y_line_manual = (-lr.coef_[0] * x_line_lr - lr.intercept_) / lr.coef_[1]

    # Визуализация
    fig=plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=df[['Personal.Loan']].to_numpy(), cmap='viridis')

    dot=scaler.transform([[ccavg_input, income_input]])

    plt.scatter(dot[0, 0], dot[0, 1], c='red', marker='x', s=1000)

    # Построение линии для вашей ручной реализации
    plt.plot(x_line_lr, y_line_manual, color='red', linewidth=2, label='Decision Boundary (manual)')

    # Добавление меток и легенды
    plt.xlabel('CCAvg')
    plt.ylabel('Income')
    plt.title('Binary Classification with Decision Boundary')
    plt.legend()
    plt.show()


    st.pyplot(fig)
df, lr, scaler ,X= load_data()
otherstuff(df, lr, scaler ,X)