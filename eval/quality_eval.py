"""
搜索拆解质量评估脚本
评估指标：
  1. 维度完整性：四维度是否全部覆盖
  2. 关键词覆盖度：每维度平均关键词数
  3. 术语命中率：预期行业术语是否出现在关键词中
  4. 模板回退率：LLM 拆解成功率 vs 回退率
"""
import json
import time
import logging
from typing import List, Dict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.search_decomposer import SearchDecomposer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_queries(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_dimension_completeness(keywords: dict, expected_dimensions: List[str]) -> float:
    """维度完整性：四维度是否全部有至少一个关键词"""
    covered = sum(1 for dim in expected_dimensions if keywords.get(dim))
    return covered / len(expected_dimensions)


def evaluate_keyword_count(keywords: dict, min_per_dim: int) -> float:
    """关键词覆盖度：每维度平均关键词数 / 最低要求"""
    counts = [len(v) for v in keywords.values() if v]
    if not counts:
        return 0.0
    avg = sum(counts) / len(counts)
    return min(avg / min_per_dim, 1.0)


def evaluate_term_hit_rate(keywords: dict, expected_terms: List[str]) -> float:
    """术语命中率：预期术语在关键词中的出现比例"""
    all_keywords = []
    for kw_list in keywords.values():
        all_keywords.extend(kw_list)
    all_text = " ".join(all_keywords).lower()
    hits = sum(1 for term in expected_terms if term.lower() in all_text)
    return hits / len(expected_terms) if expected_terms else 0.0


def run_evaluation():
    test_file = Path(__file__).parent / "test_queries.json"
    test_queries = load_test_queries(str(test_file))
    logger.info(f"Loaded {len(test_queries)} test queries")

    decomposer = SearchDecomposer()

    dim_scores = []
    kw_scores = []
    term_scores = []
    fallback_count = 0
    total_time = 0

    logger.info("=" * 80)
    logger.info("Starting evaluation...")
    logger.info("=" * 80)

    for i, query_data in enumerate(test_queries):
        topic = query_data['topic']
        expected_dimensions = query_data['expected_dimensions']
        min_keywords = query_data.get('min_keywords_per_dimension', 2)
        expected_terms = query_data.get('expected_terms', [])

        start = time.time()
        keywords = decomposer.decompose(topic)
        elapsed = time.time() - start
        total_time += elapsed

        # 检测是否回退到模板（回退的结果不包含 topic 特有词汇）
        is_fallback = keywords == decomposer._fallback(topic) or \
                      keywords == decomposer._fallback("default")
        if is_fallback:
            fallback_count += 1

        dim_score = evaluate_dimension_completeness(keywords, expected_dimensions)
        kw_score = evaluate_keyword_count(keywords, min_keywords)
        term_score = evaluate_term_hit_rate(keywords, expected_terms)
        overall = (dim_score * 0.3 + kw_score * 0.3 + term_score * 0.4)

        dim_scores.append(dim_score)
        kw_scores.append(kw_score)
        term_scores.append(term_score)

        logger.info(f"Query {i+1}: {topic}")
        logger.info(f"  Keywords: {keywords}")
        logger.info(f"  Dim={dim_score:.2f}  KW={kw_score:.2f}  Term={term_score:.2f}  "
                     f"Overall={overall:.2f}  Fallback={is_fallback}  Time={elapsed:.2f}s")
        logger.info("-" * 40)

    avg_dim = sum(dim_scores) / len(dim_scores)
    avg_kw = sum(kw_scores) / len(kw_scores)
    avg_term = sum(term_scores) / len(term_scores)
    avg_overall = (avg_dim * 0.3 + avg_kw * 0.3 + avg_term * 0.4)
    fallback_rate = fallback_count / len(test_queries)

    logger.info("=" * 80)
    logger.info("EVALUATION REPORT")
    logger.info("=" * 80)
    logger.info(f"Total queries: {len(test_queries)}")
    logger.info(f"Average dimension completeness: {avg_dim:.4f} ({avg_dim*100:.1f}%)")
    logger.info(f"Average keyword coverage:        {avg_kw:.4f} ({avg_kw*100:.1f}%)")
    logger.info(f"Average term hit rate:           {avg_term:.4f} ({avg_term*100:.1f}%)")
    logger.info(f"Overall quality score:           {avg_overall:.4f} ({avg_overall*100:.1f}%)")
    logger.info(f"Fallback rate:                   {fallback_rate:.4f} ({fallback_rate*100:.1f}%)")
    logger.info(f"Average decomposition time:      {total_time/len(test_queries):.3f}s")
    logger.info("=" * 80)

    report = {
        "total_queries": len(test_queries),
        "avg_dimension_completeness": round(avg_dim, 4),
        "avg_keyword_coverage": round(avg_kw, 4),
        "avg_term_hit_rate": round(avg_term, 4),
        "overall_quality_score": round(avg_overall, 4),
        "fallback_rate": round(fallback_rate, 4),
        "avg_decomposition_time_s": round(total_time / len(test_queries), 3),
        "details": [
            {
                "topic": test_queries[i]['topic'],
                "dim_score": round(dim_scores[i], 4),
                "kw_score": round(kw_scores[i], 4),
                "term_score": round(term_scores[i], 4),
            }
            for i in range(len(test_queries))
        ]
    }

    report_file = Path(__file__).parent / "quality_eval_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Report saved to {report_file}")

    return avg_overall, fallback_rate


if __name__ == "__main__":
    run_evaluation()
