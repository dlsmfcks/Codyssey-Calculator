import math
EPSILON = 1e-9

def mac(pattern, kernel):
    score = 0.0

    for row in range(len(pattern)):
        for col in range(len(pattern)):
            score += pattern[row][col] * kernel[row][col]
    
    if not math.isfinite(score):
        raise ValueError("MAC 점수가 실수 표현 범위를 벗어났습니다.")
    
    return score


def decide(first, second, first_label, second_label):
    if abs(first - second) < EPSILON:
        return "UNDECIDED"
    
    return first_label if first > second else second_label