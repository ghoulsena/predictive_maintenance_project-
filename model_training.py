#
# from ucimlrepo import fetch_ucirepo
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
# import matplotlib.pyplot as plt
#
# # Загрузка данных
# dataset = fetch_ucirepo(id=601)
# data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
#
# # Предобработка
# data = data.drop(columns=['UID', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
# data['Type'] = LabelEncoder().fit_transform(data['Type'])
# print("Пропущенные значения:\n", data.isnull().sum())
#
# num_cols = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
# scaler = StandardScaler()
# data[num_cols] = scaler.fit_transform(data[num_cols])
#
# X = data.drop(columns=['Machine failure'])
# y = data['Machine failure']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Создаем модели с balanced class weight
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
#     "XGBoost": XGBClassifier(
#         n_estimators=100,
#         learning_rate=0.1,
#         eval_metric='logloss',
#         random_state=42
#     ),
#     "SVM": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
# }
#
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     print(f"Модель {name} обучена.")
#
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     if hasattr(model, "predict_proba"):
#         y_pred_proba = model.predict_proba(X_test)[:, 1]
#     else:
#         if hasattr(model, "decision_function"):
#             decision_scores = model.decision_function(X_test)
#             scaler = MinMaxScaler()
#             y_pred_proba = scaler.fit_transform(decision_scores.reshape(-1, 1)).ravel()
#         else:
#             y_pred_proba = y_pred
#
#     accuracy = accuracy_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     class_report = classification_report(y_test, y_pred, zero_division=0)
#     roc_auc = roc_auc_score(y_test, y_pred_proba)
#
#     print(f"Accuracy: {accuracy:.3f}")
#     print("Confusion Matrix:")
#     print(conf_matrix)
#     print("Classification Report:")
#     print(class_report)
#     print(f"ROC-AUC: {roc_auc:.3f}")
#
#     fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#     return fpr, tpr, roc_auc
#
# plt.figure(figsize=(8,6))
# plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Random Guess')
#
# for name, model in models.items():
#     print(f"\nОценка модели: {name}")
#     fpr, tpr, roc_auc = evaluate_model(model, X_test, y_test)
#     plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
#
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC-кривые моделей')
# plt.legend()
# plt.grid()
# plt.show()
