import pandas as pd
import pytest
from src.preprocess import Preprocess, SplitConfig, PreprocessError, NUM_COLS, CAT_COLS

def test_missing_file_raises():
    with pytest.raises(PreprocessError):
        Preprocess("no_such.csv", mode = "failure")
        

def test_basic_flow_binary(tmp_path):
    # orijinal dosyayı kullanmak yerine küçük bir örnek üretelim
    sample = pd.DataFrame({
        "UDI": [1,2,3,4,5],
        "Product ID": ["M1","M2","L1","H9","L2"],
        "Type": ["M","M","L","H","L"],
        "Air temperature [K]": [298.1,298.2,298.1,298.4,298.3],
        "Process temperature [K]": [308.6,308.7,308.5,308.8,308.6],
        "Rotational speed [rpm]": [1551,1408,1498,1420,1502],
        "Torque [Nm]": [42.8,46.3,49.4,40.1,44.0],
        "Tool wear [min]": [0,3,5,2,1],
        "Target": [0,0,1,0,1],
        "Failure Type": ["No Failure","No Failure","TWF","No Failure","HDF"]
    })
    
    p = tmp_path / "toy.csv" # create a temporary file, / is used to join paths
    sample.to_csv(p, index = False)
    
    pre = Preprocess(str(p), mode = "failure", split_cfg = SplitConfig(test_size = 0.4, random_state = 0))
    X_tr, X_te, y_tr, y_te, pipeline = pre.get_preprocessed_data()
    
    out_cols = pre.get_feature_names() or  [] # get_feature_names() returns the names of the features after preprocessing
    assert not any("Failure Type" in c for c in out_cols) # Ensure that 'Failure Type' is not in the output columns
    
    #Total split
    assert len(X_tr) + len(X_te) == len(sample) # Ensure total split size matches original data
    
    assert X_tr.shape[1] >= len(NUM_COLS)
    