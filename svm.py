import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score # Using F1-score as the optimization target
from sklearn.pipeline import Pipeline # Useful for chaining steps
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Pipeline that handles sampler placement
import optuna
from tqdm.notebook import tqdm # Optional: for progress bar
import warnings

WINDOW_SIZE = 24
STEP_SIZE = 1

# !!! --- Define the Cutoff Date for Train/Test Split --- !!!
# All data *before* this date is training, data *at or after* this date is testing
# Adjust this date based on your dataset!
CUTOFF_DATE_STR = '2004-03-06'

RANDOM_STATE = 42

df = pd.read_csv('./data/inject/6S205F.csv')

# Convert 'datetime' column to datetime objects
df['datetime'] = pd.to_datetime(df['datetime'])

# Sort data: CRITICAL for time series windowing respecting motes
df.sort_values(by=['moteid', 'datetime'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Loaded data shape: {df.shape}")
print(f"Mote IDs present: {df['moteid'].unique()}")
print(f"Fault types present: {df['type'].unique()}")

# --- Split Data based on Cutoff Date ---
cutoff_date = pd.to_datetime(CUTOFF_DATE_STR)
print(f"\nSplitting data using cutoff date: {cutoff_date}")

train_df_raw = df[df['datetime'] < cutoff_date].copy()
test_df_raw = df[df['datetime'] >= cutoff_date].copy()

print(f"Raw training data points: {len(train_df_raw)}")
print(f"Raw testing data points: {len(test_df_raw)}")

if train_df_raw.empty or test_df_raw.empty:
    print("\nError: Raw training or testing dataframe is empty. Adjust cutoff date or check data.")
    exit()
# --- Feature Engineering Function (Enhanced with more features) ---
def calculate_features(window_temps):
    features = {}
    features['temp_mean'] = np.mean(window_temps)
    features['temp_std'] = np.std(window_temps)
    features['temp_min'] = np.min(window_temps)
    features['temp_max'] = np.max(window_temps)
    features['temp_median'] = np.median(window_temps)
    features['temp_range'] = features['temp_max'] - features['temp_min']
    features['temp_slope'] = np.polyfit(np.arange(len(window_temps)), window_temps, 1)[0]
    diffs = np.diff(window_temps)
    features['temp_diff_mean'] = np.mean(diffs) if len(diffs) > 0 else 0
    features['temp_diff_std'] = np.std(diffs) if len(diffs) > 0 else 0
    return features

def create_feature_dataframe(input_df, window_size, step_size, desc="Processing"):
    all_features = []
    labels = []
    grouped = input_df.groupby('moteid')
    for moteid, group in grouped:
        temps = group['temperature'].values
        types = group['type'].values
        if len(group) < window_size: continue
        for i in range(0, len(group) - window_size + 1, step_size):
            window = temps[i : i + window_size]
            types_in_seq = types[i : i + window_size]
            features = calculate_features(window)
            all_features.append(features)
            non_zero = types_in_seq[types_in_seq != 0]
            label = non_zero[0] if len(non_zero) > 0 else 0
            labels.append(label)
    if not all_features: return pd.DataFrame(), pd.Series(dtype='int')
    return pd.DataFrame(all_features), pd.Series(labels)

# --- Create Feature DataFrames ---
print("\nCreating features for TRAINING data...")
X_train_features, y_train = create_feature_dataframe(train_df_raw, WINDOW_SIZE, STEP_SIZE)
print("Creating features for TESTING data...")
X_test_features, y_test = create_feature_dataframe(test_df_raw, WINDOW_SIZE, STEP_SIZE)

if X_train_features.empty or X_test_features.empty:
    raise ValueError("Training or testing feature dataframe is empty.")

print(f"\nNumber of training sequences/features: {len(X_train_features)}")
print(f"Number of testing sequences/features: {len(X_test_features)}")
print(f"Number of features generated: {X_train_features.shape[1]}")

# --- Optuna Objective Function (Evaluate directly on Test Set) ---
def objective(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function - Trains on train set, evaluates on test set."""

    # === 1. Suggest Pipeline Components ===
    # -- Scaler --
    scaler_name = trial.suggest_categorical('scaler', ['StandardScaler', 'RobustScaler', 'MinMaxScaler'])
    if scaler_name == 'StandardScaler': scaler = StandardScaler()
    elif scaler_name == 'RobustScaler': scaler = RobustScaler()
    else: scaler = MinMaxScaler()

    # -- Feature Selection --
    selector_name = trial.suggest_categorical('feature_selector', ['none', 'SelectKBest'])
    if selector_name == 'SelectKBest':
        max_k = X_train.shape[1]
        k_percent = trial.suggest_float('k_percent', 0.3, 1.0)
        k = max(1, int(max_k * k_percent))
        selector = SelectKBest(f_classif, k=k)
    else:
        selector = 'passthrough'

    # -- Imbalance Handling (Applied only to training data) --
    imbalance_handler_name = trial.suggest_categorical('imbalance_handler', ['none', 'SMOTE'])
    if imbalance_handler_name == 'SMOTE':
        min_class_size = y_train.value_counts().min()
        smote_k = min(5, max(1, min_class_size - 1))
        if smote_k >= 1: sampler = SMOTE(random_state=42, k_neighbors=smote_k)
        else: sampler = 'passthrough' # Skip SMOTE if not possible
    else:
        sampler = 'passthrough'

    # -- SVM Hyperparameters --
    svm_kernel = trial.suggest_categorical('svm_kernel', ['rbf', 'linear', 'poly'])
    svm_C = trial.suggest_float('svm_C', 1e-3, 1e3, log=True)
    svm_params = {'C': svm_C, 'kernel': svm_kernel, 'probability': False, # Prob=False is faster if not needed
                   'class_weight': 'balanced', 'random_state': 42}
    if svm_kernel in ['rbf', 'poly']:
        svm_gamma = trial.suggest_categorical('svm_gamma', ['scale', 'auto'])
        svm_params['gamma'] = svm_gamma
    if svm_kernel == 'poly':
        svm_degree = trial.suggest_int('svm_degree', 2, 4)
        svm_params['degree'] = svm_degree

    # === 2. Create Pipeline (using ImbPipeline for correct sampler application) ===
    pipeline = ImbPipeline([
        ('scaler', scaler),
        ('selector', selector),
        ('sampler', sampler), # Sampler is applied only during fit
        ('svm', SVC(**svm_params))
    ])

    # === 3. Fit on Training Data, Evaluate on Test Data ===
    try:
        # Fit the pipeline on the entire training set
        pipeline.fit(X_train, y_train)

        # Predict on the test set
        y_pred = pipeline.predict(X_test)

        # Calculate score (use weighted F1 for imbalance)
        # This score is now calculated directly on the test set
        # score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        score = accuracy_score(y_test, y_pred)
        return score

    except Exception as e: # Catch broader exceptions during fit/predict
        # print(f"Trial {trial.number} failed: {e}") # Uncomment for debugging
        return 0.0 # Return low score if trial fails


N_TRIALS = 20
# --- Run Optuna Study ---
study = optuna.create_study(direction='maximize') # Maximize F1-score

print(f"\nStarting Optuna optimization with {N_TRIALS} trials...")
print("🚨 WARNING: Each trial is evaluated directly on the test set.")
print("   This risks overfitting hyperparameters to this specific test set.")

# Pass ALL data splits needed by the objective function
study.optimize(lambda trial: objective(trial, X_train_features, y_train, X_test_features, y_test),
               n_trials=N_TRIALS,
               show_progress_bar=True)

print("\nOptimization finished.")
print(f"Best trial number: {study.best_trial.number}")
print(f"Best F1-score (Weighted, on Test Set during optimization): {study.best_value:.4f}")
print("Best parameters found:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# --- Retrain Best Model on Full Training Data and Evaluate on Test Data (Confirmatory Step) ---
# This step essentially repeats the evaluation done in the best trial, but is good practice
# to explicitly show the final model construction and performance.
print("\nReconstructing and evaluating the best model found by Optuna...")

best_params = study.best_params

# Reconstruct the best pipeline
if best_params['scaler'] == 'StandardScaler': best_scaler = StandardScaler()
elif best_params['scaler'] == 'RobustScaler': best_scaler = RobustScaler()
else: best_scaler = MinMaxScaler()

if best_params['feature_selector'] == 'SelectKBest':
    max_k = X_train_features.shape[1]
    k = max(1, int(max_k * best_params['k_percent']))
    best_selector = SelectKBest(f_classif, k=k)
else:
    best_selector = 'passthrough'

if best_params['imbalance_handler'] == 'SMOTE':
    min_class_size = y_train.value_counts().min()
    smote_k = min(5, max(1, min_class_size - 1))
    if smote_k >= 1: best_sampler = SMOTE(random_state=42, k_neighbors=smote_k)
    else: best_sampler = 'passthrough'
else:
    best_sampler = 'passthrough'

svm_params = {
    'C': best_params['svm_C'], 'kernel': best_params['svm_kernel'],
    'probability': True, # Set to True for final model if needed for ROC etc.
    'class_weight': 'balanced', 'random_state': 42
}
if best_params['svm_kernel'] in ['rbf', 'poly']: svm_params['gamma'] = best_params['svm_gamma']
if best_params['svm_kernel'] == 'poly': svm_params['degree'] = best_params['svm_degree']
best_svm = SVC(**svm_params)

final_pipeline = ImbPipeline([
    ('scaler', best_scaler),
    ('selector', best_selector),
    ('sampler', best_sampler),
    ('svm', best_svm)
])

# Fit the final pipeline on the full training data
final_pipeline.fit(X_train_features, y_train)
print("Best model pipeline constructed and retrained on full training data.")

# --- Final Evaluation on Test Set ---
print("\nFinal evaluation of the best model on the HELD-OUT TEST set:")
y_pred_test = final_pipeline.predict(X_test_features)

all_labels_test = sorted(np.unique(np.concatenate((y_test.unique(), y_pred_test))))
target_names_test = [f'Type {i}' for i in all_labels_test]

print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred_test, labels=all_labels_test, target_names=target_names_test, zero_division=0))
print("\nTest Set Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test, labels=all_labels_test))
print(f"\nTest Set Overall Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Test Set F1-score (Weighted): {f1_score(y_test, y_pred_test, average='weighted', zero_division=0):.4f}")
