
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def normalized_rmse(y_true, y_pred):
    """
    Calculate Normalized RMSE based on the given formula.
    
    Args:
    y_true (array-like): True IC50 values.
    y_pred (array-like): Predicted IC50 values.
    
    Returns:
    float: Normalized RMSE.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    normalized_rmse = rmse / (np.max(y_true) - np.min(y_true))
    return normalized_rmse

def correct_ratio(y_true, y_pred):
    """
    Calculate Correct Ratio where Absolute Error in pIC50 is <= 0.5.
    
    Args:
    y_true (array-like): True IC50 values.
    y_pred (array-like): Predicted IC50 values.
    
    Returns:
    float: Correct Ratio.
    """
    pIC50_true = -np.log10(y_true * 1e-9)
    pIC50_pred = -np.log10(y_pred * 1e-9)
    
    absolute_errors = np.abs(pIC50_true - pIC50_pred)
    correct_ratio = np.mean(absolute_errors <= 0.5)
    return correct_ratio

def calculate_score(y_true, y_pred):
    """
    Calculate the final score based on the provided formula.
    
    Args:
    y_true (array-like): True IC50 values.
    y_pred (array-like): Predicted IC50 values.
    
    Returns:
    float: Final score.
    """
    A = normalized_rmse(y_true, y_pred)
    B = correct_ratio(y_true, y_pred)
    
    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    return score
    
# 40% public, 60% private score
def evaluate_scores(y_true, y_pred):
    """
    Calculate public and private scores based on a split of the test dataset.
    
    Args:
    y_true (array-like): True IC50 values for the entire test dataset.
    y_pred (array-like): Predicted IC50 values for the entire test dataset.
    
    Returns:
    tuple: public_score, private_score
    """
    # Test 데이터셋을 40%와 60%로 나누기
    y_true_public, y_true_private, y_pred_public, y_pred_private = train_test_split(
        y_true, y_pred, test_size=0.6, random_state=42
    )
    
    # Public score 계산 (40%)
    public_score = calculate_score(y_true_public, y_pred_public)
    
    # Private score 계산 (60%)
    private_score = calculate_score(y_true_private, y_pred_private)
    
    return public_score, private_score

# 예시 사용법:
# y_true_test = np.array([...])  # 실제 test 데이터셋의 IC50 값
# y_pred_test = np.array([...])  # 모델이 예측한 test 데이터셋의 IC50 값

# public_score, private_score = evaluate_scores(y_true_test, y_pred_test)

# print(f"Public Score: {public_score}")
# print(f"Private Score: {private_score}")