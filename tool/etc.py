import numpy as np

def pearson_correlation(x, y):
    # 평균 계산
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # 편차 계산
    xm = x - mean_x
    ym = y - mean_y
    
    # 상관계수 계산
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm**2) * np.sum(ym**2)) + 1e-8
    r = r_num / r_den
    
    return r

def pIC50_to_IC50(pIC50): # IC50 -> nM 단위
    return 10**(-pIC50+9)