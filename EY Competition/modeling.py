import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

# Try importing LightGBM, handle if missing
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    warnings.warn("LightGBM not found. The ensemble will skip it.")
    HAS_LGB = False

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold, cross_validate
import config

# -----------------------------------------------------------------------------
# 1. STACKING ENSEMBLE
# -----------------------------------------------------------------------------
def get_hybrid_ensemble():
    """
    Returns a Stacking Regressor:
    1. XGBoost (Trees)
    2. LightGBM (Trees)
    3. MLP (Neural Net)
    Meta-learner: Ridge Regression
    """
    
    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        tree_method='hist',
        random_state=42
    )

    # --- Neural Network ---
    # Wrapped in pipeline to ensure scaling and imputation
    nn_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle any remaining NaNs
        ('scaler', StandardScaler()), 
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=64,
            learning_rate='adaptive',
            max_iter=500, # Lower iter, let early stopping handle it
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ))
    ])

    estimators = [('xgb', xgb_model), ('nn', nn_pipeline)]

    # --- LightGBM ---
    if HAS_LGB:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=31,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
        estimators.append(('lgb', lgb_model))
    

    # --- The Stack ---
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(), 
        cv=5,
        n_jobs=-1,
        passthrough=False 
    )

    return stack

# -----------------------------------------------------------------------------
# 2. LOG-TRANSFORM
# -----------------------------------------------------------------------------
def build_robust_pipeline():
    """
    Wraps the ensemble in a Log-Transformer to handle skewed water quality data.
    """
    base_model = get_hybrid_ensemble()
    
    model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,        # y -> log(y+1)
        inverse_func=np.expm1 # pred -> exp(pred)-1
    )
    return model

# -----------------------------------------------------------------------------
# 3. EVALUATION ENGINE (Spatial CV)
# -----------------------------------------------------------------------------
def evaluate_model(model, X, y, groups, target_name):
    """
    Performs Group K-Fold Cross Validation (Spatial Split).
    This prevents the model from 'cheating' by seeing data from the same location.
    """
    print(f"\nEvaluating architecture on {target_name}...")
    
    # GroupKFold ensures no location is in both Train and Test sets
    gkf = GroupKFold(n_splits=5)
    
    scoring = {
        'r2': 'r2',
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }
    
    results = cross_validate(
        model, X, y, groups=groups, cv=gkf, 
        scoring=scoring, n_jobs=-1, return_train_score=False
    )
    
    mean_r2 = np.mean(results['test_r2'])
    mean_rmse = -np.mean(results['test_rmse'])
    mean_mae = -np.mean(results['test_mae'])
    
    print(f"   5-Fold Spatial CV Results:")
    print(f"      R^2:   {mean_r2:.4f} (±{np.std(results['test_r2']):.3f})")
    print(f"      RMSE: {mean_rmse:.4f}")
    
    return mean_r2, mean_rmse

# -----------------------------------------------------------------------------
# 4. MAIN ENTRY POINT
# -----------------------------------------------------------------------------
def train_models(df):
    """
    Called by main.py. Orchestrates the training.
    """
    # 1. Feature Selection
    drop_cols = config.TARGETS + ['Sample Date', 'Latitude', 'Longitude', 'Year', 'Month', 'key']
    features = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Features selected ({len(features)}): {features}")

    # 2. Loop Targets
    performance_report = {}

    for target in config.TARGETS:
        if target not in df.columns: continue
            
        print(f"\n{'='*40}")
        print(f"TARGET: {target}")
        print(f"{'='*40}")
        
        # 3. Prepare Data
        target_df = df.dropna(subset=[target])
        
        # Define Groups (Location ID) for Spatial CV
        groups = (target_df['Latitude'].round(4).astype(str) + "_" + 
                  target_df['Longitude'].round(4).astype(str))
        
        X = target_df[features]
        y = target_df[target]
        
        # Optional: Clip extreme sensor errors
        q99 = y.quantile(0.99)
        mask = y <= q99
        X = X[mask]
        y = y[mask]
        groups = groups[mask]
        
        # Safety: Drop any rows with NaN values
        nan_mask = ~X.isna().any(axis=1)
        if not nan_mask.all():
            n_dropped = (~nan_mask).sum()
            print(f"   Dropping {n_dropped} rows with NaN values...")
            X = X[nan_mask]
            y = y[nan_mask]
            groups = groups[nan_mask]
        
        print(f"   Training Samples: {len(X)}")

        # 4. Train & Evaluate
        model = build_robust_pipeline()
        r2, rmse = evaluate_model(model, X, y, groups, target)
        performance_report[target] = {'R2': r2, 'RMSE': rmse}
        
        # 5. Final Fit (on all data)
        print("   Retraining final model on full dataset...")
        model.fit(X, y)

    print("\n" + "="*40)
    print(" FINAL RESULTS (Spatial CV)")
    print("="*40)
    for t, m in performance_report.items():
        print(f"{t}: R^2={m['R2']:.3f}, RMSE={m['RMSE']:.3f}")
