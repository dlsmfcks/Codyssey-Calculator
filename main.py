import json
import math
import re
from pathlib import Path
from time import perf_counter

from npu import EPSILON, decide, mac

REPEATS = 100
JSON_SIZES = (5, 13, 25)
DATA_FILE = Path(__file__).with_name("data.json")

def timecomplexity(samples):
    print(f"\n[성능 분석] 필터 1개 기준, {REPEATS}회 평균")
    print("크기       평균 시간(ms)    MAC 횟수(N²)")
    print("------------------------------------------")

    for pattern, kernel in samples:
        size = len(pattern)

        start = perf_counter()

        for _ in range(REPEATS):
            mac(pattern, kernel)

        elapsed = perf_counter() - start

        mean_ms = elapsed * 1000 / REPEATS

        print(f"{size:2}×{size:<2}      {mean_ms:10.6f}    {size * size:5}")


def read_matrix(name):
    print(f"\n{name} 입력 (3줄, 한 줄에 숫자 3개)")
    matrix = []
    while len(matrix) < 3:
        tokens = input(f"{len(matrix) + 1}행: ").split()
        try:
            row = [float(token) for token in tokens]
            if len(row) == 3 and all(math.isfinite(v) for v in row):
                matrix.append(row)
                continue
        except ValueError:
            pass
        print("입력 형식 오류. 같은 행에 유한한 숫자 3개를 공백으로 구분해 입력하세요.")
    return matrix

def check_matrix(matrix, size, name):
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError(f"{name}의 행 수가 {size}여야 합니다.")
    for index, row in enumerate(matrix, 1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"{name}의 {index}행에는 값이 {size}개 필요합니다.")

        if any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise ValueError(f"{name}의 {index}행에 유효하지 않는 숫자가 존재합니다.")
        
    return [[float(v) for v in row] for row in matrix]


def normalize_label(label):
    labels = {"+": "Cross", "cross": "Cross", "x": "X"}
    if isinstance(label, str) and label.strip().lower() in labels:
        return labels[label.strip().lower()]
    raise ValueError(f"알 수 없는 라벨 {label!r}")


def load_filters(raw, size):
    group = raw.get(f"size_{size}") if isinstance(raw, dict) else None

    if not isinstance(group, dict):
        raise ValueError(
            f"size_{size} 필터 묶음이 없거나 형식이 잘못되었습니다."
        )

    filters = {}

    for key, matrix in group.items():
        label = normalize_label(key)

        if label in filters:
            raise ValueError(f"{label} 필터가 중복되었습니다.")

        filters[label] = check_matrix(matrix, size, label)

    if set(filters) != {"Cross", "X"}:
        raise ValueError("Cross, X 필터가 각각 하나씩 필요합니다.")

    return filters


def analyze_case(key, case, filters, filter_errors):
    match = re.fullmatch(r"size_(5|13|25)_([0-9]+)", key)
    if not match:
        raise ValueError("패턴 키는 size_{5 또는 13 또는 25}_{번호} 형식이어야 합니다.")
    size = int(match.group(1))
    if size in filter_errors:
        raise ValueError(filter_errors[size])
    if not isinstance(case, dict) or "input" not in case or "expected" not in case:
        raise ValueError("패턴에 input과 expected가 필요합니다.")
        
    pattern = check_matrix(case["input"], size, "패턴")
    expected = normalize_label(case["expected"])
    cross = mac(pattern, filters[size]["Cross"])
    x = mac(pattern, filters[size]["X"])
    result = decide(cross, x, "Cross", "X")
    print(f"  Cross 점수 {cross} | X 점수 {x} | 판정 {result} | expected {expected}")
    if result == "UNDECIDED":
        return f"두 점수의 차이가 {EPSILON:g} 미만이어서 판정 불가"
    if result != expected:
        return f"점수가 더 높은 필터({result})와 정답({expected})이 다름"
    
    return None


def manual_mode():
    first = read_matrix("필터 A")
    second = read_matrix("필터 B")
    print("필터 A, B 저장 완료")
    pattern = read_matrix("패턴")

    a = mac(pattern, first)
    b = mac(pattern, second)
    result = decide(a, b, "A", "B")
    print(f"\nA 점수 {a} | B 점수 {b}")
    print("판정:", "판정 불가 (UNDECIDED)" if result == "UNDECIDED" else result)
    timecomplexity([(pattern, first)])


def json_mode():
    with DATA_FILE.open(encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("patterns"), dict):
        raise ValueError("patterns가 케이스 키를 가진 JSON 객체여야 합니다.")
    
    filters, filter_errors = {}, {}
    for size in JSON_SIZES:
        try:
            filters[size] = load_filters(data.get("filters"), size)
            print(f"size_{size} 필터 로드 완료 (Cross, X)")
        except (ValueError, OverflowError) as error:
            filter_errors[size] = str(error)
            print(f"size_{size} 필터 오류 | {error}")
    
    patterns = data["patterns"]
    failures = {}
    for key, case in patterns.items():
        print(f"\n[{key}]")
        try:
            reason = analyze_case(key, case, filters, filter_errors)
        except (ValueError, OverflowError) as error:
            reason = str(error)
        if reason:
            failures[key] = reason
            print(f"  FAIL | {reason}")
        else:
            print("  PASS")

    samples = []
    for size in (3, *JSON_SIZES):
        cross = [[float(r == size // 2 or c == size // 2)
                    for c in range(size)] for r in range(size)]
        samples.append((cross, cross))
    timecomplexity(samples)
    
    print(f"\n[결과 요약] 총 {len(patterns)}개 | 통과 {len(patterns) - len(failures)}개 | 실패 {len(failures)}개")
    for key, reason in failures.items():
        print(f"- {key} | {reason}")



def main():
    print("\n==========================================================")
    print("                      Mini NPU Simulator                    ")
    print("==========================================================")

    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("==========================================================")
    
    while True:
        choice = input("선택: ").strip()
        if choice == "1":
            manual_mode()
            return
        elif choice == "2":
            json_mode()
            return
        print("1 또는 2를 입력하세요.")

if __name__ == "__main__":
    try:
        main()
    except EOFError:
        print("\n입력이 끝났습니다. 모드 1에서는 각 행렬에 3줄씩 입력하세요.")
    except (OSError, ValueError) as error:
        print(f"실행 오류 | {error}")