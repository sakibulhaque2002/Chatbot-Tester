from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import json
import requests

# Adapters
from adapters.cosine_adapter import CosineAdapter
from adapters.bertscore_adapter import BERTScoreAdapter
from adapters.gemini_adapter import GeminiAdapter

app = FastAPI()

@app.post("/test")
async def run_tests(
    chatbot_api_url: str = Form(...),
    similarity_metric: str = Form(...),
    test_cases_file: UploadFile = File(...)
):
    # ---------------------
    # Load test cases
    # ---------------------
    try:
        test_cases = json.load(test_cases_file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")

    # ---------------------
    # Instantiate adapter
    # ---------------------
    metric = similarity_metric.lower()
    if metric == "cosine":
        adapter = CosineAdapter()
    elif metric == "bertscore":
        adapter = BERTScoreAdapter()
    elif metric == "gemini":
        adapter = GeminiAdapter()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported similarity metric: {similarity_metric}")

    # ---------------------
    # Run tests
    # ---------------------
    results = []
    passed = 0
    failed = 0

    for case in test_cases:
        query = case.get("query", "")
        expected = case.get("expected", "")

        # Call chatbot API
        try:
            resp = requests.post(chatbot_api_url, json={"query": query}, timeout=10).json()
            actual = resp.get("answer", "")
        except Exception as e:
            actual = ""
            failed += 1
            results.append({
                "query": query,
                "expected": expected,
                "actual": actual,
                "similarity": 0,
                "status": "FAIL",
                "error": str(e)
            })
            continue

        # Compute similarity
        similarity = adapter.compute(actual, expected)
        status = "PASS" if adapter.check_pass(actual, expected) else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        results.append({
            "query": query,
            "expected": expected,
            "actual": actual,
            "similarity": round(similarity, 3),
            "status": status
        })

    total = len(test_cases)
    accuracy = round((passed / total) * 100, 2) if total > 0 else 0

    return JSONResponse({
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "results": results
    })
