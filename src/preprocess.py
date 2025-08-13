from __future__ import annotations # It allows for forward references in type hints, which is useful for type annotations that refer to classes not yet defined.
from dataclasses import dataclass # It is used to define classes that are primarily used to store data.
from typing import List, Tuple, Optional, Literal, Dict, Any # It provides support for type hints, which can be used to indicate the expected types of variables, function parameters, and return values.
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer # It is used to apply different preprocessing steps to different columns of a dataset.
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline # It is used to create a sequence of data processing steps, allowing for a streamlined workflow in machine learning tasks.

TargetMode = Literal["failure", "failure_type"]

NUM_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CAT_COLS = [
    "Type" # "L/M/H"
]

DROP_ALWAYS = ["UDI", "Product ID"]
COL_TARGET_BINARY = "Target"
COL_TARGET_MULTI = "Failure Type"


class PreprocessError(Exception):
    """Custom exception for preprocessing errors."""
    pass

@dataclass
class SplitConfig:
    """Configuration for train-test split."""
    test_size: float = 0.2 
    random_state: int = 42
    stratify: bool = True # If True, it ensures that the split maintains the same proportion of classes in both training and testing sets.
    

class Preprocess:
    """
    Class for preprocessing the dataset.
    
    Parameters
    ----------
    data_path : str
        Path to CSV file containing the dataset.
    mode : {'failure', 'failure_type'}
        Target mode: binary (failure) or multi-calss (failure_type)
    split_cfg : SplitConfig
        Configuration for train-test split.
    """
    
    def __init__(self, data_path:str, mode: TargetMode, split_cfg: SplitConfig = SplitConfig()) -> None: # None is the default return type for functions that do not return a value.
        if not os.path.exists(data_path):
            raise PreprocessError(f"Data filepath is not found: {data_path}")
        
        if mode not in ("failure", "failure_type"):
            raise PreprocessError(f"Invalid mode: {mode}")
        
        self.data_path = data_path
        self.mode = mode
        self.split_cfg = split_cfg
        self._pipeline: Optional[Pipeline] = None # It is used to indicate that the variable may not be initialized immediately and will be set later.
        self._feature_names_out: Optional[List[str]] = None # It is used to indicate that the variable may not be initialized immediately and will be set later.
        
    def get_preprocessed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
        """_summary_
        Reads CSV, selects appropriate columns, prevents data leakage, builds a preprocessing pipeline and applies fit_transform / transform methods to the data. After train/test split, returns X_train, X_test, y_train, y_test and fitted preprocessing pipeline. 
        """
    
        df = self._load() # Load the dataset from the CSV file.
        df = self._basic_clean(df) # Perform basic cleaning of the dataset.
        df = self._feature_engineer(df) 
        X, y = self._select_xy(df) # Select features and target variable.
        pipe = self._build_pipeline(X) # Build the preprocessing pipeline.
        
        
        #Stratify
        strat = y if self.split_cfg.stratify else None # It is used to ensure that the split maintains the same proportion of classes in both training and testing sets.
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size = self.split_cfg.test_size,
            random_state = self.split_cfg.random_state,
            stratify = strat
        )
        
        #Fit pipeline on train, transform train and test data
        X_tr_t = pd.DataFrame(pipe.fit_transform(X_tr))
        X_te_t = pd.DataFrame(pipe.transform(X_te))
        
        # Set feature names after transformation
        try:
            self._feature_names_out = pipe.get_feature_names_out().tolist()
            X_tr_t.columns = self._feature_names_out
            X_te_t.columns = self._feature_names_out
        except Exception:
            pass
        
        self._pipeline = pipe
        
        return X_tr_t, X_te_t, y_tr.reset_index(drop = True), y_te.reset_index(drop = True), pipe
    
    def get_feature_names(self) -> Optional[List[str]]:
        """Returns the feature names after preprocessing."""
        return self._feature_names_out
    
    def get_pipeline(self) -> Pipeline:
        """_summary_
        Returns fitted or not fitted preprocessing pipeline.
        """
        if self._pipeline is None:
            raise PreprocessError("Pipeline is not built yet. Call get_preprocessed_data() first.")
        
        return self._pipeline
    
    def get_processed_data(self):
        """Alias for get_preprocessed_data()."""
        return self.get_preprocessed_data()
    
    # ---------- internals ----------
    
    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.data_path)
            expected = set(NUM_COLS + CAT_COLS + [COL_TARGET_BINARY, COL_TARGET_MULTI] + DROP_ALWAYS) # It is used to ensure that the dataset contains all the expected columns.
            missing = [c for c in expected if c not in df.columns]
            if missing:
                raise PreprocessError(f"Missing columns in the dataset: {missing}")
            
            return df
        except Exception as e:
            raise PreprocessError(f"Error loading data: {e}") from e
        
    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: dropping columns, type control, basic missing value handling."""
        try:
            # Drop columns that are not needed for analysis
            df = df.drop(columns = [c for c in DROP_ALWAYS if c in df.columns])
            
            # Missing values: numeric -> median, categorical -> mode
            for c in NUM_COLS:
                if df[c].isna().any(): # It checks if there are any missing values in the column.
                    df[c] = df[c].fillna(df[c].median())
                
            for c in CAT_COLS:
                if df[c].isna().any():
                    mode_val = df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown" # It retrieves the most frequent value in the column, or "Unknown" if the mode is empty.
                    df[c] = df[c].fillna(mode_val)
            
            # Unexpected NaN value control in target columns
            if self.mode == "failure":
                if df[COL_TARGET_BINARY].isna().any():
                    raise PreprocessError("Missing values in Target.")
            elif self.mode == "failure_type":
                if df[COL_TARGET_MULTI].isna().any():
                    raise PreprocessError("Missing values in Failure Type.")

            
            return df
        except Exception as e:
            raise PreprocessError(f"Error during basic cleaning: {e}") from e
    
    def _select_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """ Selecting X/y depending on the mode and preventing data leakage"""
        try:
            if self.mode == "failure":
                drop_cols = [COL_TARGET_MULTI]
                y = df[COL_TARGET_BINARY].astype(int)
            else:
                drop_cols = [COL_TARGET_BINARY]
                y = df[COL_TARGET_MULTI]
        
            X = df.drop(columns = drop_cols + [COL_TARGET_BINARY, COL_TARGET_MULTI])
            keep_cols = [c for c in NUM_COLS + CAT_COLS if c in X.columns]
            X = X[keep_cols].copy()  # It creates a copy of the DataFrame to avoid modifying the original data.
            
            return X, y
    
        except Exception as e:
            raise PreprocessError (f"Error selecting X/y: {e}") from e
        
    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """ A sklearn pipeline containging preprocessing steps for numeric and categorical features."""
        try:
            num_cols = X.select_dtypes(include="number").columns.tolist()
            cat_cols = X.select_dtypes(exclude="number").columns.tolist()
            
            transformers = []
            if num_cols:
                transformers.append(("num", StandardScaler(), num_cols)) # It standardizes the numeric features by removing the mean and scaling to unit variance.
            if cat_cols:
                transformers.append(("cat", OneHotEncoder(handle_unknown = "ignore"), cat_cols)) # It encodes categorical features as a one-hot numeric array, ignoring any unknown categories.
                
            pre = ColumnTransformer(
                transformers = transformers,
                remainder = "drop",
            ) # It applies the specified transformations to the selected columns and drops any remaining columns that are not specified in the transformers.
            
            pipe = Pipeline(steps = [("pre", pre)]) # It creates a pipeline with the preprocessing step.
            
            return pipe
        except Exception as e:
            raise PreprocessError(f"Error building preprocessing pipeline: {e}") from e
        
    def _feature_engineer(self,df: pd.DataFrame) -> pd.DataFrame:
        """Domain-specific simple features drived from EDA"""
        try:
            #Temp_Diff
            df["Temp_Diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
            
            #Torque_Speed_Ratio
            denom = df["Rortational speed [rpm]"].replace(0, pd.NA) # It replaces 0 with NaN to avoid division by zero.
            df["Torque_Speed_Ratio"] = (df["Torque [Nm]"] / denom).fillna(0.0) # It calculates the ratio of torque to rotational speed, filling NaN values with 0.0.
            return df
        except Exception as e:
            raise PreprocessError(f"Error during feature engineering: {e}") from e
            