from myai_rag.query_rewrite import rewrite_retrieval_query


ALIASES = {
    "滨江消费品": "滨江消费品有限公司",
    "滨江消费品有限公司": "滨江消费品有限公司",
    "澜赋科技": "澜赋科技有限公司",
    "澜赋科技有限公司": "澜赋科技有限公司",
    "蓝天旅游": "蓝天旅游有限公司",
    "蓝天旅游有限公司": "蓝天旅游有限公司",
}


def rewrite(question, history):
    return rewrite_retrieval_query(question, history, ALIASES)


def test_standalone_query_is_unchanged():
    result = rewrite("滨江消费品2021年营业收入是多少？", [])
    assert result.rewritten_query == "滨江消费品2021年营业收入是多少？"
    assert result.used_history is False


def test_follow_up_inherits_company_year_and_preserves_current_intent():
    result = rewrite(
        "那净利润呢？",
        [{"role": "user", "content": "滨江消费品2021年营业收入是多少？"}],
    )
    assert result.rewritten_query == "滨江消费品有限公司2021年净利润是多少？"
    assert result.inherited_companies == ("滨江消费品有限公司",)
    assert result.inherited_years == ("2021",)


def test_explicit_new_company_does_not_inherit_old_company():
    result = rewrite(
        "澜赋科技2021年净利润是多少？",
        [{"role": "user", "content": "滨江消费品2021年营业收入是多少？"}],
    )
    assert result.rewritten_query == "澜赋科技2021年净利润是多少？"
    assert result.used_history is False


def test_short_standalone_query_with_explicit_company_does_not_inherit_old_topic():
    result = rewrite(
        "澜赋科技员工人数？",
        [{"role": "user", "content": "滨江消费品2021年营业收入是多少？"}],
    )
    assert result.rewritten_query == "澜赋科技员工人数？"
    assert result.used_history is False


def test_comparison_inherits_previous_company_and_topic():
    result = rewrite(
        "和澜赋科技相比呢？",
        [{"role": "user", "content": "滨江消费品2021年总负债是多少？"}],
    )
    assert "滨江消费品有限公司" in result.rewritten_query
    assert "澜赋科技有限公司" in result.rewritten_query
    assert "2021年" in result.rewritten_query
    assert "总负债" in result.rewritten_query
    assert "相比如何" in result.rewritten_query


def test_unit_conversion_inherits_metric():
    result = rewrite(
        "换算成万元呢？",
        [{"role": "user", "content": "蓝天旅游2021年净利润是多少？"}],
    )
    assert result.rewritten_query == "蓝天旅游有限公司2021年净利润换算成万元是多少？"


def test_ambiguous_reference_requests_clarification():
    result = rewrite("那它呢？", [])
    assert result.needs_clarification is True
    assert "公司名称" in result.clarification_question


def test_assistant_answer_is_not_used_as_retrieval_context():
    result = rewrite(
        "那净利润呢？",
        [
            {"role": "assistant", "content": "滨江消费品2021年的营业收入是100亿元。"},
        ],
    )
    assert result.needs_clarification is True
    assert "100亿元" not in result.rewritten_query


def test_only_recent_user_context_is_used():
    history = [
        {"role": "user", "content": "滨江消费品2020年净利润是多少？"},
        {"role": "user", "content": "滨江消费品2021年净利润是多少？"},
    ]
    result = rewrite("换算成万元呢？", history)
    assert "2021年" in result.rewritten_query
    assert "2020年" not in result.rewritten_query


def test_slots_can_be_resolved_from_more_than_one_recent_user_turn():
    history = [
        {"role": "user", "content": "滨江消费品营业收入是多少？"},
        {"role": "user", "content": "看2021年的。"},
    ]
    result = rewrite("那净利润呢？", history)
    assert result.rewritten_query == "滨江消费品有限公司2021年净利润是多少？"
