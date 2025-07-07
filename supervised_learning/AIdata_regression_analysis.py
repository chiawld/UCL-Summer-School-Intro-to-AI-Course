import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the clean data
df = pd.read_csv('/Users/chicswldrg/Desktop/Summer School Juli 2025/UCL-Summer-School-Intro-to-AI-Course-1/datasets/ai_adoption_dataset_cleaned.csv')

print(f"Data loaded successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Look at the data
print("\nFirst few rows:")
print(df.head())

######################## CHOOSE WHAT TO PREDICT & USE ########################################################################################################

# Look at the columns to decide what to predict
print("Available columns in our dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col} - {df[col].dtype}")

print("\nNumerical columns (good for prediction targets):")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(numerical_cols)

print("\nAll columns (can be used as features):")
all_cols = df.columns.tolist()
print(all_cols)
print('\n')

####### CHOOSE TARGET AND FEATURES

# What we want to predict (must be numerical)
target_column = 'adoption_rate'  # Change this to your target column

# What we'll use to make predictions (choose 2-5 columns)
feature_columns = ['daily_active_users', 'year']  # Change these

print(f"We want to predict: {target_column}")
print(f"Using these features: {feature_columns}")
print('\n')

################################# CHECK VALIDITY OF CHOICE ################################################################################################

def check_validity_choices():
    if target_column not in df.columns:
        print(f"ERROR: {target_column} not found in data")
    else:
        print(f"✓ Target column '{target_column}' found")

def check_missing_ft():
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"ERROR: These features not found: {missing_features}")
    else:
        print("✓ All feature columns found")

check_validity_choices()
check_missing_ft()

############################### PREPARE DATA FOR MACHINE LEARNING ################################################################################################
y = df[target_column]  # What we want to predict
x = df[feature_columns]  # What we'll use to predict

def prepare_data(x, y):
    # Create target (Y) and features (X)

    print("Data before preparation:")
    print(f"Target shape: {y.shape}")
    print(f"Features shape: {x.shape}")
    print(f"Features data types:\n{x.dtypes}")

    # Convert text columns to numbers (one-hot encoding)
    x_prepared = pd.get_dummies(x, drop_first=True)

    print(f"\nData after preparation:")
    print(f"Features shape: {x_prepared.shape}")
    print(f"New column names: {list(x_prepared.columns)}")

    # Remove any rows with missing values
    mask = ~(x_prepared.isnull().any(axis=1) | y.isnull())
    x_clean = x_prepared[mask]
    y_clean = y[mask]

    print(f"\nAfter removing missing values:")
    print(f"Final dataset size: {len(x_clean)} rows")
    print(f"Features: {x_clean.shape[1]} columns")

    return x_clean, y_clean

# Identify categorical and numerical features
categorical_features = x.select_dtypes(include=['object']).columns.tolist()
numerical_features = x.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# CHOOSE YOUR ENCODING METHOD HERE
# Option 1: One-hot encoding (creates separate columns for each category)
# Option 2: Label encoding (assigns numbers to categories)

ENCODING_METHOD = "one_hot"  # Change to "label" for label encoding

print(f"\n🔧 Using {ENCODING_METHOD} encoding for categorical features...")

if ENCODING_METHOD == "one_hot":
    # One-hot encoding: Good when you have few categories per feature
    X_encoded = pd.get_dummies(x, columns=categorical_features, drop_first=True)
    print(f"✓ One-hot encoding applied")
    print(f"  Original features: {x.shape[1]}")
    print(f"  After encoding: {X_encoded.shape[1]} features")
    
elif ENCODING_METHOD == "label":
    # Label encoding: Good when you have many categories or ordinal data
    X_encoded = x.copy()
    label_encoders = {}
    
    for feature in categorical_features:
        le = LabelEncoder()
        X_encoded[feature] = le.fit_transform(X[feature].astype(str))
        label_encoders[feature] = le
        print(f"  {feature}: {len(le.classes_)} categories → 0 to {len(le.classes_)-1}")
    
    print(f"✓ Label encoding applied")
    print(f"  Features remain: {X_encoded.shape[1]} (same as original)")

print(f"\nNew feature names: {list(X_encoded.columns)}")

x_clean, y_clean = prepare_data(x, y)

######################## SPLIT DATA INTO TRAINING AND TEST SETS ################################################################################################

x_train, x_test, y_train, y_test = train_test_split(
    x_clean, y_clean, 
    test_size=0.2,      # Use 20% for testing
    random_state=42     # For reproducible results
)

print("Data split successfully!")
print(f"Training set: {x_train.shape[0]} rows")
print(f"Testing set: {x_test.shape[0]} rows")
print(f"Features: {x_train.shape[1]} columns")

print(f"\nTarget variable summary:")
print(f"Training target range: {y_train.min():.0f} to {y_train.max():.0f}")
print(f"Testing target range: {y_test.min():.0f} to {y_test.max():.0f}")

print('\n')

######################## TRAIN REGRESSION MODELS ################################################################################################

############## Model 1: Linear Regression (draws a straight line through the data) ###############################################################
print("Training Model 1: Linear Regression\n")
model1 = LinearRegression()
model1.fit(x_train, y_train)

# Make predictions
y_pred1 = model1.predict(x_test)

# Calculate performance
mse1 = mean_squared_error(y_test, y_pred1)
r2_1 = r2_score(y_test, y_pred1)

print(f"✓ Linear Regression (Ridge) trained!")
print(f"  Mean Squared Error: {mse1:.2f}")
print(f"  R² Score: {r2_1:.3f} (higher is better, max = 1.0)")

############### Model 2: Decision Tree (makes decisions like a flowchart) ######################################################################
print("\nTraining Model 2: Decision Tree\n")
model2 = DecisionTreeRegressor(random_state=42, max_depth=10)
model2.fit(x_train, y_train)

# Make predictions
y_pred2 = model2.predict(x_test)

# Calculate performance
mse2 = mean_squared_error(y_test, y_pred2)
r2_2 = r2_score(y_test, y_pred2)

print(f"✓ Decision Tree trained!")
print(f"  Mean Squared Error: {mse2:.2f}")
print(f"  R² Score: {r2_2:.3f} (higher is better, max = 1.0)")

############### Model 3: Random Forest (combines many decision trees) ######################################################################
print("\n Training Model 3: Random Forest\n")
model3 = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model3.fit(x_train, y_train)

# Make predictions
y_pred3 = model3.predict(x_test)

# Calculate performance
mse3 = mean_squared_error(y_test, y_pred3)
r2_3 = r2_score(y_test, y_pred3)

print(f"✓ Random Forest trained!")
print(f"  Mean Squared Error: {mse3:.2f}")
print(f"  R² Score: {r2_3:.3f} (higher is better, max = 1.0)\n")

######################## COMPARE MODEL PERFORMANCE ################################################################################################
# Compare all models
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
    'Mean_Squared_Error': [mse1, mse2, mse3],
    'R2_Score': [r2_1, r2_2, r2_3]
})

print("🏆 MODEL COMPARISON RESULTS")
print("=" * 50)
print(results.to_string(index=False, float_format='%.3f'))

# Find the best model
best_model_idx = results['R2_Score'].idxmax()
best_model_name = results.loc[best_model_idx, 'Model']
best_r2 = results.loc[best_model_idx, 'R2_Score']

print(f"\n🥇 WINNER: {best_model_name}")
print(f"   R² Score: {best_r2:.3f}")

print(f"\nHow to interpret R² Score:")
print(f"- 1.0 = Perfect predictions")
print(f"- 0.8+ = Very good")
print(f"- 0.6+ = Good") 
print(f"- 0.4+ = Okay")
print(f"- Below 0.4 = Needs improvement\n")

################# VISUALIZE MODEL PERFORMANCE ################################################################################################
# Plot actual vs predicted values for each model
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = [
    ('Linear Regression', y_pred1),
    ('Decision Tree', y_pred2), 
    ('Random Forest', y_pred3)
]

for i, (name, predictions) in enumerate(models):
    axes[i].scatter(y_test, predictions, alpha=0.6)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[i].set_xlabel('Actual Values')
    axes[i].set_ylabel('Predicted Values')
    axes[i].set_title(f'{name}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("How to read these plots:")
print("- Points close to the red line = good predictions")
print("- Points far from the red line = poor predictions")
print("- Tighter cluster around the line = better model")

# Show which features are most important (for Random Forest)
print("🔍 FEATURE IMPORTANCE (Random Forest)")
print("This shows which features matter most for predictions:")
print("=" * 60)

feature_importance = pd.DataFrame({
    'Feature': x_clean.columns,
    'Importance': model3.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False, float_format='%.3f'))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance['Importance'])
plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

########################## MAKE PREDICTIONS ON NEW DATA ################################################################################################
# Use the best model to make predictions
models = [model1, model2, model3]
best_model = models[best_model_idx]

print(f"Using {best_model_name} for predictions")

# Example: Make predictions on a few test cases
print("\n📊 SAMPLE PREDICTIONS")
print("=" * 50)

# Show first 5 predictions vs actual values
sample_size = min(5, len(x_test))
sample_predictions = best_model.predict(x_test.iloc[:sample_size])

for i in range(sample_size):
    actual = y_test.iloc[i]
    predicted = sample_predictions[i]
    error = abs(actual - predicted)
    
    print(f"Case {i+1}:")
    print(f"  Actual: {actual:.0f}")
    print(f"  Predicted: {predicted:.0f}")
    print(f"  Error: {error:.0f}")
    print()

print("💡 TIP: You can use this model to predict new values!")
print("Just create a new row with the same features and use model.predict()")

print('\n')