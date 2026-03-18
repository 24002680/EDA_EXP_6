# EDA_EXP_6

**Aim**

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal.

**Algorithm**

1)Import pandas, numpy, seaborn, matplotlib, sklearn libraries.

**Program**

Your Name GUNASUNDARI B
Your Reg No.212224040093

    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    
    1 - DATA UNDERSTANDING
    print("First 5 rows:\n", df.head())
    print("\nDataset shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    
    2 - UNIVARIATE ANALYSIS (HISTPLOTTING)
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    sns.histplot(df['alcohol'], kde=True)
    plt.title("Alcohol Distribution")
    
    plt.subplot(1,3,2)
    sns.histplot(df['volatile acidity'], kde=True)
    plt.title("Volatile Acidity Distribution")
    
    plt.subplot(1,3,3)
    sns.histplot(df['pH'], kde=True)
    plt.title("pH Distribution")
    
    plt.tight_layout()
    plt.show()
    
    
    3 - BIVARIATE ANALYSIS
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    sns.boxplot(x='quality', y='alcohol', data=df)
    plt.title("Alcohol vs Quality")
    
    plt.subplot(1,2,2)
    sns.boxplot(x='quality', y='volatile acidity', data=df)
    plt.title("Acidity vs Quality")
    
    plt.tight_layout()
    plt.show()
    
    print("\nRelationship Explanation:")
    print("- Higher quality wines tend to have higher alcohol levels.")
    print("- Volatile acidity decreases as wine quality increases.")
    
    
    
    4 - MULTIVARIATE ANALYSIS – CORRELATION
    
    corr = df[['alcohol', 'volatile acidity', 'pH', 'quality']].corr()
    
    plt.figure(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
    
    print("\nHighest Correlation with Quality:")
    print(corr['quality'].sort_values(ascending=False))
    
    
    
    5 - CLASSIFICATION – GOOD VS BAD WINE
    
    df['good_wine'] = (df['quality'] >= 7).astype(int)
    
    X = df.drop(['quality', 'good_wine'], axis=1)
    y = df['good_wine']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    
    6 - OUTLIER DETECTION
    
    features = ['alcohol', 'pH', 'volatile acidity']
    
    plt.figure(figsize=(12, 4))
    
    for i, feature in enumerate(features, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(y=df[feature])
        plt.title(f"{feature} Boxplot")
    
    plt.tight_layout()
    plt.show()


**Output**


<img width="669" height="671" alt="564839977-68b727bd-b9fd-489e-b455-2d4a6ea16e04" src="https://github.com/user-attachments/assets/01d3174b-83ef-42c8-896c-216b85c779e6" />




<img width="1487" height="400" alt="564840138-02814ab0-0283-4f87-9b43-4fab9dfb541c" src="https://github.com/user-attachments/assets/3f28a798-7f5c-4922-954b-15052d7d2ec2" />


<img width="1189" height="506" alt="564840256-d876db32-7daf-4c45-90e4-49555a07037e" src="https://github.com/user-attachments/assets/7f354b36-4584-467c-9385-a73356830e55" />

<img width="618" height="658" alt="564840349-e0253718-a612-4a42-9b44-3315fc0aada2" src="https://github.com/user-attachments/assets/e0f853d7-dd1c-4057-b9ae-b1d1577d1961" />


<img width="1214" height="398" alt="564840431-0aa2643d-75a5-4b29-890a-0542a6f7f77c" src="https://github.com/user-attachments/assets/3d3ee9bf-9a77-41d5-aa5c-16f71e237fc0" />



**Result**
Thus, To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal has successfully completed.
