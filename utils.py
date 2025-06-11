import re
from collections import Counter

def remove_duplicates(dict_list):
    """
    Loại bỏ các dict trùng lặp dựa trên giá trị của key 'title'.
    Giữ lại lần xuất hiện đầu tiên của mỗi title.
    
    Args:
        dict_list (list of dict): Danh sách các dict có chứa key 'title'.
    
    Returns:
        list of dict: Danh sách đã loại bỏ các dict có 'title' trùng.
    """
    seen_titles = set()
    unique_list = []
    for item in dict_list:
        title = item.get('title')
        if title not in seen_titles:
            seen_titles.add(title)
            unique_list.append(item)
    return unique_list

def extract_answer(text: str) -> str:
    marker = "answer is:"
    parts = text.split(marker, 1)
    if len(parts) > 1:
        # Lấy phần sau marker, rồi strip bớt khoảng trắng đầu/cuối
        return parts[1].strip()
    return ""

def tokenize(s: str) -> list[str]:
    """
    Lowercase, strip leading/trailing spaces, and split on whitespace.
    """
    return s.lower().strip().split()

def extract_titles(dict_list: list[dict[str, any]]) -> list[str]:
    """
    Nhận một list các dict và trả về list các giá trị của key 'title'.

    Args:
        dict_list: List các dict, mỗi dict có thể chứa key 'title'.

    Returns:
        List[str]: Danh sách các giá trị title (nếu dict không có 'title', trả về None hoặc giá trị mặc định).
    """
    titles: list[str] = []
    for item in dict_list:
        titles.append(item.get('title'))
    return titles

from collections import Counter
from typing import List

def cal_f1_score(answer: List[str], target: List[str]) -> float:
    """
    Tính F1 score giữa hai list string, không quan tâm thứ tự,
    và xét đầy đủ số lần xuất hiện (multiset).

    F1 = 2 * (precision * recall) / (precision + recall)
    với:
      precision = matched_count / len(answer)
      recall    = matched_count / len(target)

    Args:
        answer:   list dự đoán
        target:   list ground-truth

    Returns:
        F1 score
    """
    # lowercase tất cả
    ans_lower = [a.lower() for a in answer]
    tgt_lower = [t.lower() for t in target]

    cnt_ans = Counter(ans_lower)
    cnt_tgt = Counter(tgt_lower)

    # matched_count = tổng min(counts) cho mỗi token
    matched = cnt_ans & cnt_tgt
    matched_count = sum(matched.values())

    # precision, recall
    if not answer and not target:
        return 1.0
    if not answer or not target:
        return 0.0

    precision = matched_count / len(answer)
    recall    = matched_count / len(target)

    if precision + recall == 0:
        return 0.0
    return precision, recall, 2 * (precision * recall) / (precision + recall)


def cal_accuracy_score(answer: List[str], target: List[str]) -> float:
    """
    Tính accuracy giữa hai list string, không quan tâm thứ tự,
    xét multiset. Định nghĩa:
    
      accuracy = matched_count / len(target)
    
    (tức phần tử ground-truth được cover bao nhiêu phần)

    Args:
        answer: list dự đoán
        target: list ground-truth

    Returns:
        accuracy
    """
    # lowercase
    ans_lower = [a.lower() for a in answer]
    tgt_lower = [t.lower() for t in target]

    cnt_ans = Counter(ans_lower)
    cnt_tgt = Counter(tgt_lower)

    matched = cnt_ans & cnt_tgt
    matched_count = sum(matched.values())

    if not target:
        return 1.0 if not answer else 0.0

    return matched_count / len(target)


# Ví dụ test
if __name__ == "__main__":
    preds = ["hello", "world", "test", "OpenAI"]
    gts   = ["hello", "planet", "test", "openai"]

    f1   = exact_match_f1(preds, gts)
    acc  = accuracy_score(preds, gts)
    print(f"Exact-match F1 score: {f1:.3f}")   # 0.500
    print(f"Accuracy score:      {acc:.3f}")  # 0.500




# Example usage:
if __name__ == "__main__":
    # Example lists of generated answers and reference answers
    generated_answers = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "The RAG model retrieves documents and uses them to generate responses."
    ]
    reference_answers = [
        "Paris",
        "Artificial intelligence includes the subset of machine learning.",
        "RAG retrieves documents and then generates responses using them."
    ]

    f1_list, average_f1 = evaluate_rag_f1(generated_answers, reference_answers)
    print("F1 Scores for each example:", f1_list)
    print("Average F1 Score:", average_f1)
