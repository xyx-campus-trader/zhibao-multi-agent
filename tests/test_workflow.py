"""
工作流单元测试
"""
import pytest
from core.search_decomposer import SearchDecomposer


def test_decomposer_fallback():
    decomposer = SearchDecomposer()
    result = decomposer._fallback("AI")
    assert "政策动向" in result
    assert len(result["政策动向"]) > 0


def test_decomposer_default():
    decomposer = SearchDecomposer()
    result = decomposer._fallback("未知行业")
    assert "政策动向" in result
    assert result == decomposer._fallback("anything")


def test_regex_extract():
    decomposer = SearchDecomposer()
    text = '{"政策动向": ["AI政策", "监管"], "公司动态": ["产品发布"]}'
    result = decomposer._regex_extract(text)
    assert isinstance(result, dict)
