import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Abhi@6366",
    database="user_engagement"
)

query="SELECT * FROM users"
df=pd.read_sql(query,conn)

print(df.head())

print(df.info())

avg_usage=df.groupby("Plan")["UsageHours"].mean()
print("\nAverage Usage By Plan: ")
print(avg_usage)
plt.figure(figsize=(8,5))
avg_usage.plot(kind="bar",color=["#3498db", "#e74c3c", "#f1c40f"],edgecolor="black")
plt.title("Average Usage by Subscription Plan ",fontsize=12)
plt.xticks(rotation=0)
plt.ylabel("Average Usage Hours")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


region_usage=df.groupby("Region")["UsageHours"].mean()
print("\nAverage Usage By Region: ")
print(region_usage)

plt.figure(figsize=(8, 5))
region_usage.plot(kind="bar",color="#2ecc71",edgecolor="black")
plt.title("Average Usage By Region",fontsize=14)
plt.xlabel("Region",fontsize=12)
plt.ylabel("Average Usage Hours",fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

def retention_category(hours):

    if hours>=70:
        return "Highly Active"
    elif hours >=30:
        return "Moderately Active"
    else:
        return "Low Engagement"
    
df["RetentionCategory"]=df["UsageHours"].apply(retention_category) 
print("\nUser Retention Categories:")
print(df[["UserID","UsageHours","RetentionCategory"]])  

retention_counts=df["RetentionCategory"].value_counts()
print("\nRetention Category Counts:")
print(retention_counts)
plt.figure(figsize=(7,7))
retention_counts.plot(kind="pie",autopct="%1.1f%%",startangle=90,
    colors=["#2ecc71", "#f1c40f", "#e74c3c"],
    wedgeprops={"edgecolor": "black"})
plt.title("User retention Distribution",fontsize=14)
plt.ylabel("")
plt.show()

top_users=df.sort_values(by="UsageHours",ascending=False).head(5)
print("\nTop 5 Most Active Users: ")
print(top_users[["UserID","UsageHours","Plan"]])

plt.figure(figsize=(8,5))
plt.barh(top_users["UserID"],top_users["UsageHours"],   color="#9b59b6",edgecolor="black")
plt.title("Top 5 Most Active Users",fontsize=14)
plt.xlabel("Usage Hours",fontsize=12)
plt.ylabel("User ID",fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

churn_users=df[df["UsageHours"]<10]
print("\nPotential Churn Risk Users:")
print(churn_users[["UserID","UsageHours","Plan"]])

churn_count=churn_users["Plan"].value_counts()
plt.figure(figsize=(8,5))
churn_count.plot(kind="bar", color="#e74c3c",edgecolor="black")
plt.title("Churn Risk Users by Plan",fontsize=14)
plt.xlabel("Plan",fontsize=12)
plt.ylabel("Number of Users",fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


total_users=len(df)

avg_usage=df["UsageHours"].mean()

highly_active=len(df[df["RetentionCategory"] == "Highly Active"])

churn_risk=len(churn_users)

print("\nMaximum Usage Hours:")
print(df["UsageHours"].max())

print("\nRetention Category Counts:")
print(df["RetentionCategory"].value_counts())

print("\n====KPI METRICS =====")

print(f"Total Users:{total_users}")

print(f"Average Usage Hours:{avg_usage:.2f}")

print(f"Highly Active Users:{highly_active}")

print(f"Churn Risk Users:{churn_risk}")

correlation=df["LoginCount"].corr(df["UsageHours"])

print("\nCorrelation between Login Count and Usage hours")
print(correlation)

plt.figure(figsize=(8,5))

plt.scatter(df["LoginCount"],df["UsageHours"], color="#27ae60",edgecolor="black", alpha=0.7)
plt.title("Login Count vs Usage Hours",fontsize=14)
plt.xlabel("Login Count",fontsize=12)
plt.ylabel("Usage Hours",fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Machine Learning Model
X=df[["LoginCount",'UsageHours']]

y=df["RetentionCategory"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print("\nModel Accuracy:")
print(accuracy)

new_user = pd.DataFrame(
    [[45,75]],
    columns=["LoginCount","UsageHours"]
)
prediction=model.predict(new_user)

print("\nPrediction for new user :")
print(prediction)

print("\n========== FINAL BUSINESS INSIGHTS ==========")

print("""
1. Pro users show the highest engagement and usage hours.

2. Users with low login count generally show low platform activity.

3. High login frequency is strongly related to higher engagement.

4. Some users show churn-risk behavior due to very low usage.

5. Machine Learning model successfully predicts user engagement categories.
""")



conn.close()

