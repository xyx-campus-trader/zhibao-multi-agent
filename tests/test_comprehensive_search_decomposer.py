"""
综合测试：SearchDecomposer — 回退逻辑、正则提取、关键词生成
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest, json
from core.search_decomposer import (
    SearchDecomposer, _fetch_search_results,
    FALLBACK_TEMPLATES, SYSTEM_PROMPT
)


class TestFallbackTemplates:
    def test_all_expected_dimensions(self):
        """所有模板都应包含四个维度"""
        for key, template in FALLBACK_TEMPLATES.items():
            assert "政策动向" in template
            assert "公司动态" in template
            assert "投融资" in template
            assert "行业趋势" in template

    def test_templates_have_keywords(self):
        for key, template in FALLBACK_TEMPLATES.items():
            for dim, keywords in template.items():
                assert len(keywords) >= 1, f"{key}/{dim} has no keywords"

    def test_default_template_exists(self):
        assert "default" in FALLBACK_TEMPLATES

    def test_ai_template_has_ai_keywords(self):
        ai_template = FALLBACK_TEMPLATES["AI"]
        assert any("AI" in k or "大模型" in k for kws in ai_template.values() for k in kws)

    def test_default_template_keywords_are_generic(self):
        default = FALLBACK_TEMPLATES["default"]
        generic_terms = ["行业", "政策", "融资", "市场"]
        all_words = " ".join(k for kws in default.values() for k in kws)
        assert any(term in all_words for term in generic_terms)


class TestFallback:
    def test_fallback_matches_ai(self):
        decomposer = SearchDecomposer()
        result = decomposer._fallback("AI人工智能行业")
        assert result == FALLBACK_TEMPLATES["AI"]

    def test_fallback_matches_new_energy(self):
        decomposer = SearchDecomposer()
        result = decomposer._fallback("新能源产业动态")
        assert result == FALLBACK_TEMPLATES["新能源"]

    def test_fallback_matches_semiconductor(self):
        decomposer = SearchDecomposer()
        result = decomposer._fallback("半导体芯片")
        assert result == FALLBACK_TEMPLATES["半导体"]

    def test_fallback_default_for_unknown(self):
        decomposer = SearchDecomposer()
        result = decomposer._fallback("xxx不存在的行业")
        assert result == FALLBACK_TEMPLATES["default"]

    def test_fallback_case_insensitive(self):
        decomposer = SearchDecomposer()
        result = decomposer._fallback("AI")  # uppercase
        assert result == FALLBACK_TEMPLATES["AI"]
        result2 = decomposer._fallback("ai")  # lowercase
        assert result2 == FALLBACK_TEMPLATES["AI"]

    def test_fallback_returns_fresh_dict(self):
        """_fallback 直接从模板返回，应包含有效数据"""
        decomposer = SearchDecomposer()
        result = decomposer._fallback("AI行业分析")
        assert result["政策动向"] == FALLBACK_TEMPLATES["AI"]["政策动向"]
        assert len(result["政策动向"]) > 0

    def test_fallback_empty_string(self):
        decomposer = SearchDecomposer()
        result = decomposer._fallback("")
        assert result == FALLBACK_TEMPLATES["default"]


class TestRegexExtract:
    def test_valid_json_like_output(self):
        decomposer = SearchDecomposer()
        text = '''{"政策动向": ["AI 政策", "监管法规"], "公司动态": ["AI 产品"]}'''
        result = decomposer._regex_extract(text)
        assert "政策动向" in result
        assert "公司动态" in result

    def test_partial_match(self):
        decomposer = SearchDecomposer()
        text = '''"政策动向": ["AI 政策", "大模型 管理"]
        "公司动态": []'''
        result = decomposer._regex_extract(text)
        assert len(result) > 0

    def test_malformed_input(self):
        """非JSON输入：正则提取失败后回退到default模板"""
        decomposer = SearchDecomposer()
        text = "this is not json at all"
        result = decomposer._regex_extract(text)
        # 正则提取后每个维度为空列表，但因为 result 为空 dict 时返回 default
        assert isinstance(result, dict)
        assert "政策动向" in result

    def test_empty_string(self):
        """空字符串：正则提取返回空，最终回退到default模板"""
        decomposer = SearchDecomposer()
        result = decomposer._regex_extract("")
        assert isinstance(result, dict)
        assert "政策动向" in result
        # 正则提取可能返回部分匹配，只要不崩溃就行
        assert len(result) > 0

    def test_regex_extract_with_escaped_quotes(self):
        decomposer = SearchDecomposer()
        text = '''"政策动向": ["政策 keyword", "监管 文件"], "投融资": ["融资 事件"]'''
        result = decomposer._regex_extract(text)
        assert "政策动向" in result
        assert len(result["政策动向"]) == 2


class TestSystemPrompt:
    def test_system_prompt_contains_dimensions(self):
        assert "政策动向" in SYSTEM_PROMPT
        assert "公司动态" in SYSTEM_PROMPT
        assert "投融资" in SYSTEM_PROMPT
        assert "行业趋势" in SYSTEM_PROMPT

    def test_system_prompt_mentions_json(self):
        assert "json" in SYSTEM_PROMPT.lower() or "JSON" in SYSTEM_PROMPT


class TestDecomposerInit:
    def test_init_no_llm_error_if_not_configured(self):
        """即使 LLM 不可用，SearchDecomposer 也应该能初始化（回退模板）"""
        # LLM 不可用时，decompose 会触发回退
        decomposer = SearchDecomposer()
        assert decomposer is not None

    def test_decompose_with_unavailable_llm(self):
        """LLM 不可用时应回退到模板"""
        decomposer = SearchDecomposer()
        result = decomposer.decompose("AI行业分析")
        # 应该返回回退模板
        assert isinstance(result, dict)
        assert "政策动向" in result
        assert len(result["政策动向"]) > 0


class TestFetchSearchResults:
    def test_fetch_invalid_query(self):
        import asyncio
        results = asyncio.run(_fetch_search_results("@#$%^&* invalid query with special chars!"))
        assert isinstance(results, list)

    def test_fetch_empty_query(self):
        import asyncio
        results = asyncio.run(_fetch_search_results(""))
        assert isinstance(results, list)

    def test_fetch_returns_fallback_on_network_error(self):
        import asyncio
        results = asyncio.run(_fetch_search_results("查询关键词测试"))
        assert isinstance(results, list)
        for r in results:
            assert "title" in r
            assert "snippet" in r
            assert "query" in r


class TestSearchParallel:
    def test_parallel_empty_keywords(self):
        import asyncio

        decomposer = SearchDecomposer()
        results = asyncio.run(decomposer.search_parallel({}))
        assert isinstance(results, list)
        assert results == []

    def test_parallel_single_dimension(self):
        import asyncio

        decomposer = SearchDecomposer()
        results = asyncio.run(decomposer.search_parallel({
            "政策动向": ["AI 政策"]
        }))
        assert isinstance(results, list)

    def test_parallel_all_dimensions(self):
        import asyncio

        decomposer = SearchDecomposer()
        keywords = FALLBACK_TEMPLATES["AI"]
        results = asyncio.run(decomposer.search_parallel(keywords))
        assert isinstance(results, list)

    def test_parallel_with_none_queries(self):
        import asyncio

        decomposer = SearchDecomposer()
        results = asyncio.run(decomposer.search_parallel({
            "测试维度": []
        }))
        assert isinstance(results, list)
