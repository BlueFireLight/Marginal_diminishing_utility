import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # Hides minor scikit-learn warnings

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv('Dataset Grp2.csv')

# Calculate Total Production (Output 'Y')
df['Rice_Production'] = df['Rice_Area (million hectare)'] * df['Rice_Yield (Kg./Hectare)']
df['Wheat_Production'] = df['Wheat_Area (million hectare)'] * df['Wheat_Yield (Kg./Hectare)']

# Log Transformations - Targets
df['ln_Rice_Y'] = np.log(df['Rice_Production'])
df['ln_Wheat_Y'] = np.log(df['Wheat_Production'])

# Log Transformations - Common Features
df['ln_Fert'] = np.log(df['Fertilizer (lakh tonnes)'])
df['ln_Pest'] = np.log(df['Pesticides (thousand tonnes)'])
df['ln_Seeds'] = np.log(df['Seeds (lakh quintals)'])
df['ln_Elec'] = np.log(df['Electricity used for Irrigation(GWH)'])
df['ln_Total_Elec'] = np.log(df['Electricity_percapita(KWh)'])

# Log Transformations - Crop Specific Features
df['ln_Rice_Area'] = np.log(df['Rice_Area (million hectare)'])
df['ln_Wheat_Area'] = np.log(df['Wheat_Area (million hectare)'])

# USING REAL MSP (Adjusted for Inflation)
df['ln_MSP_Rice'] = np.log(df['MSP_adjusted_rice'])
df['ln_MSP_Wheat'] = np.log(df['MSP_adjusted_wheat'])

# Define EXACTLY the 7 feature sets
rice_features = ['ln_Rice_Area', 'ln_Fert', 'ln_Pest', 'ln_MSP_Rice', 'ln_Seeds', 'ln_Elec', 'ln_Total_Elec']
wheat_features = ['ln_Wheat_Area', 'ln_Fert', 'ln_Pest', 'ln_MSP_Wheat', 'ln_Seeds', 'ln_Elec', 'ln_Total_Elec']
feature_names_display = ['Area', 'Fertilizer', 'Pesticides', 'Real MSP', 'Seeds', 'Irrig_Electricity', 'Total_Electricity']

# Train-Test Split (Chronological)
train_df = df[df['Year'] <= 2011].copy()
test_df = df[df['Year'] >= 2012].copy()

# ==========================================
# 2. SCALING (MEAN-CENTERING)
# ==========================================
def process_centered_data(train, test, feature_cols):
    means = train[feature_cols].mean()
    train_centered = train.copy()
    test_centered = test.copy()
    for c in feature_cols:
        train_centered[c] = train[c] - means[c]
        test_centered[c] = test[c] - means[c]
    return train_centered, test_centered

train_rice_c, test_rice_c = process_centered_data(train_df, test_df, rice_features)
train_wheat_c, test_wheat_c = process_centered_data(train_df, test_df, wheat_features)
# ==========================================
# 3. TRANSLOG GENERATOR
# ==========================================
def get_translog_features(data, feature_cols):
    df_tl = pd.DataFrame(index=data.index)
    for c in feature_cols:
        df_tl[c] = data[c]
    for c in feature_cols:
        df_tl[f"{c}_sq"] = 0.5 * (data[c] ** 2)
    return df_tl

# ==========================================
# 4. TRAINING & EVALUATION ENGINE
# ==========================================
def train_and_evaluate(crop_name, train_c, test_c, train_raw, test_raw, feature_cols, y_col, y_actual):
    tscv = TimeSeriesSplit(n_splits=5)

    # --- COBB-DOUGLAS ---
    cd_model = LinearRegression()
    cd_model.fit(train_c[feature_cols], train_raw[y_col])
    cd_preds = np.exp(cd_model.predict(test_c[feature_cols]))
    cd_acc = 100 - (mean_absolute_percentage_error(test_raw[y_actual], cd_preds) * 100)
    cd_A = np.exp(cd_model.intercept_)
    cd_elas = dict(zip(feature_names_display, cd_model.coef_))

    # --- TRANSLOG (Ridge) ---
    X_train_tl = get_translog_features(train_c, feature_cols)
    X_test_tl = get_translog_features(test_c, feature_cols)
    tl_model = RidgeCV(alphas=np.logspace(-2, 4, 100), cv=tscv)
    tl_model.fit(X_train_tl, train_raw[y_col])
    tl_preds = np.exp(tl_model.predict(X_test_tl))
    tl_acc = 100 - (mean_absolute_percentage_error(test_raw[y_actual], tl_preds) * 100)
    tl_A = np.exp(tl_model.intercept_)
    tl_elas = dict(zip(feature_names_display, tl_model.coef_[:len(feature_cols)]))

    # --- PRINT TABLES ---
    print(f"\n{'='*80}\n{crop_name.upper()} PRODUCTION MODELS\n{'='*80}")
    print(f"{'Metric':<25} | {'Cobb-Douglas':<20} | {'Translog (Ridge)':<20}")
    print("-" * 80)
    print(f"{'Output Accuracy (2012-23)':<25} | {cd_acc:>19.2f}% | {tl_acc:>19.2f}%")
    print(f"{'Total Factor Prod (A)':<25} | {cd_A:>19.0f} | {tl_A:>19.0f}")
    print("-" * 80)
    for k in cd_elas.keys():
        print(f"Elasticity: {k:<17} | {cd_elas[k]:>19.4f} | {tl_elas[k]:>19.4f}")

    return cd_preds, tl_preds
# ==========================================
# 5. RUN MODELS & SAVE PREDICTIONS
# ==========================================
rice_cd_preds, rice_tl_preds = train_and_evaluate(
    'Rice', train_rice_c, test_rice_c, train_df, test_df, rice_features, 'ln_Rice_Y', 'Rice_Production')

wheat_cd_preds, wheat_tl_preds = train_and_evaluate(
    'Wheat', train_wheat_c, test_wheat_c, train_df, test_df, wheat_features, 'ln_Wheat_Y', 'Wheat_Production')

test_df['Rice_Pred_CD'] = rice_cd_preds
test_df['Rice_Pred_TL'] = rice_tl_preds
test_df['Wheat_Pred_CD'] = wheat_cd_preds
test_df['Wheat_Pred_TL'] = wheat_tl_preds

print("\n\n=== RICE: ACTUAL VS PREDICTED (Test Set: 2012-2023) ===")
print(test_df[['Year', 'Rice_Production', 'Rice_Pred_CD', 'Rice_Pred_TL']].to_string(index=False, float_format="%.0f"))

print("\n=== WHEAT: ACTUAL VS PREDICTED (Test Set: 2012-2023) ===")
print(test_df[['Year', 'Wheat_Production', 'Wheat_Pred_CD', 'Wheat_Pred_TL']].to_string(index=False, float_format="%.0f"))
# ==========================================
# 5. RUN MODELS & SAVE PREDICTIONS
# ==========================================
rice_cd_preds, rice_tl_preds = train_and_evaluate(
    'Rice', train_rice_c, test_rice_c, train_df, test_df, rice_features, 'ln_Rice_Y', 'Rice_Production')

wheat_cd_preds, wheat_tl_preds = train_and_evaluate(
    'Wheat', train_wheat_c, test_wheat_c, train_df, test_df, wheat_features, 'ln_Wheat_Y', 'Wheat_Production')

test_df['Rice_Pred_CD'] = rice_cd_preds
test_df['Rice_Pred_TL'] = rice_tl_preds
test_df['Wheat_Pred_CD'] = wheat_cd_preds
test_df['Wheat_Pred_TL'] = wheat_tl_preds

print("\n\n=== RICE: ACTUAL VS PREDICTED (Test Set: 2012-2023) ===")
print(test_df[['Year', 'Rice_Production', 'Rice_Pred_CD', 'Rice_Pred_TL']].to_string(index=False, float_format="%.0f"))

print("\n=== WHEAT: ACTUAL VS PREDICTED (Test Set: 2012-2023) ===")
print(test_df[['Year', 'Wheat_Production', 'Wheat_Pred_CD', 'Wheat_Pred_TL']].to_string(index=False, float_format="%.0f"))
import matplotlib.pyplot as plt

# ==========================================
# 6. PLOT ACTUAL VS PREDICTED (COBB-DOUGLAS & TRANSLOG)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Increased figure size for more lines

# --- Rice Production Plot ---
axes[0].plot(test_df['Year'], test_df['Rice_Production'], marker='o', label='Actual Production', color='blue', linewidth=2)
axes[0].plot(test_df['Year'], test_df['Rice_Pred_CD'], marker='s', linestyle='--', label='Cobb-Douglas Predicted', color='orange', linewidth=2)
axes[0].plot(test_df['Year'], test_df['Rice_Pred_TL'], marker='^', linestyle=':', label='Translog Predicted', color='red', linewidth=2)
axes[0].set_title('Rice Production: Actual vs Predicted (2012-2023)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Total Production', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, linestyle=':', alpha=0.7)

# --- Wheat Production Plot ---
axes[1].plot(test_df['Year'], test_df['Wheat_Production'], marker='o', label='Actual Production', color='green', linewidth=2)
axes[1].plot(test_df['Year'], test_df['Wheat_Pred_CD'], marker='s', linestyle='--', label='Cobb-Douglas Predicted', color='purple', linewidth=2)
axes[1].plot(test_df['Year'], test_df['Wheat_Pred_TL'], marker='^', linestyle=':', label='Translog Predicted', color='darkblue', linewidth=2)
axes[1].set_title('Wheat Production: Actual vs Predicted (2012-2023)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_ylabel('Total Production', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('Cobb_Douglas_and_Translog_Prediction.png', bbox_inches='tight') # Changed filename to reflect both models
plt.show()
