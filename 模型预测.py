# 高血糖风险预警模型 - （模型构建与训练）
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, roc_curve)
import joblib

# ====================== 1. 加载清洗后的标准化数据 ======================
# 方式1：本地读取（与diabetes_cleaned.csv同目录）
df = pd.read_csv('diabetes_cleaned.csv')

# 方式2：云端读取（团队协作，解决路径不一致问题，来自中期报告GitHub链接）
# df = pd.read_csv('https://raw.githubusercontent.com/qieqiesha/hyperglycemia-risk-alert/main/diabetes_cleaned.csv')

# 分离特征与标签
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# ====================== 2. 分层抽样划分训练集/测试集（8:2） ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # 分层抽样，保证正负样本比例一致
    random_state=42  # 固定随机种子，结果可复现
)
print(f"训练集样本数：{X_train.shape[0]}，测试集样本数：{X_test.shape[0]}")
print(f"训练集患病比例：{y_train.mean():.3f}，测试集患病比例：{y_test.mean():.3f}")

# ====================== 3. 定义模型（中期报告指定3种模型） ======================
# 模型1：逻辑回归（基线模型）
lr = LogisticRegression(max_iter=1000, random_state=42)

# 模型2：随机森林（主模型）
rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=6, 
    random_state=42, 
    class_weight='balanced'  # 解决样本不均衡，提升高风险检出率
)

# 模型3：XGBoost（对比模型）
xgb = XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    random_state=42,
    use_label_encoder=False, 
    eval_metric='logloss'
)

# 模型字典，统一训练与评估
models = {
    "逻辑回归(基线)": lr,
    "随机森林(主模型)": rf,
    "XGBoost(对比模型)": xgb
}

# ====================== 4. 模型训练与评估 ======================
# 存储评估结果
model_results = []

print("="*80)
print("高血糖风险预警模型 - 评估结果（核心指标：AUC、召回率Recall）")
print("="*80)

for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算核心指标
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # 高风险人群检出率
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 保存结果
    model_results.append([name, acc, precision, recall, f1, auc])
    
    # 打印评估报告
    print(f"\n📌 模型：{name}")
    print(f"准确率(Accuracy)：{acc:.4f}")
    print(f"精确率(Precision)：{precision:.4f}")
    print(f"召回率(Recall,高风险检出率)：{recall:.4f}")
    print(f"F1分数：{f1:.4f}")
    print(f"AUC值：{auc:.4f}")
    print("-"*60)
    print(classification_report(y_test, y_pred, target_names=["健康(0)", "高血糖(1)"]))

# ====================== 5. 模型性能对比表 ======================
result_df = pd.DataFrame(
    model_results, 
    columns=["模型", "准确率", "精确率", "召回率", "F1", "AUC"]
).sort_values(by=["AUC", "召回率"], ascending=False)

print("\n" + "="*80)
print("📊 多模型性能对比排名（按AUC→召回率降序）")
print("="*80)
print(result_df.round(4))

# ====================== 6. 保存最优模型（供邵梓懿做可解释性分析） ======================
best_model_name = result_df.iloc[0]["模型"]
best_model = models[best_model_name]
joblib.dump(best_model, "best_hyperglycemia_model.pkl")
print(f"\n✅ 最优模型【{best_model_name}】已保存，可用于SHAP可解释性分析！")

# ====================== 7. 输出特征重要性（主模型-随机森林） ======================
print("\n" + "="*80)
print("🔍 随机森林主模型 - 特征重要性排序")
print("="*80)
feature_importance = pd.DataFrame({
    "特征": X.columns,
    "重要性": rf.feature_importances_
}).sort_values("重要性", ascending=False)
print(feature_importance.round(4))

# ====================== 程序结束不关闭窗口（按回车退出） ======================
input("\n🎉 程序全部执行完毕！按 回车键 关闭窗口...")