import json
from pathlib import Path

# 与后端配置一致的路径
BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_DB_PATH = BASE_DIR / "feedback_db.json"
FEW_SHOT_PATH = BASE_DIR / "few_shot_examples.json"

def update_few_shot():
    # 读取所有反馈
    with open(FEEDBACK_DB_PATH, "r", encoding="utf-8") as f:
        feedbacks = json.load(f)
    
    # 筛选优质示例
    good_examples = [
        {
            "question": f["question"],
            "context": f["context"],
            "answer": f["answer"],
            "timestamp": f.get("timestamp", 0),
        }
        for f in feedbacks
        if f["feedback"] == "useful" and len(f["context"]) > 100  # 确保上下文完整
    ]
    
    # 去重（按问题保留时间戳最新的一条）
    latest_by_question = {}
    for ex in good_examples:
        key = "".join(ex["question"].lower().split())
        previous = latest_by_question.get(key)
        if previous is None or ex["timestamp"] > previous["timestamp"]:
            latest_by_question[key] = ex
    unique_examples = list(latest_by_question.values())
    
    # 保留最新5条（按时间戳排序）
    unique_examples.sort(key=lambda x: -x.get("timestamp", 0))
    final_examples = [
        {
            "question": ex["question"],
            "context": ex["context"],
            "answer": ex["answer"],
        }
        for ex in unique_examples[:5]
    ]
    
    # 保存到示例文件
    with open(FEW_SHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_examples, f, ensure_ascii=False, indent=2)
    print(f"更新完成，共 {len(final_examples)} 条示例")

if __name__ == "__main__":
    update_few_shot()
