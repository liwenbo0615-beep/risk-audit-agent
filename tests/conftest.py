import os

import pytest

# 测试默认离线、确定性、不联网。注意：不再全局设 AUTO_REVIEW_DECISION=skip，
# 否则会掩盖"人工复核被静默跳过"的问题——需要跳过的测试自行显式传 auto_decision。
os.environ["OFFLINE_DEMO_MODE"] = "1"


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch, tmp_path):
    """每个测试前后重置依赖环境变量的模块级单例，避免跨测试串味。

    graph 也一起重置，避免节点/路由编排变化时测试间复用旧 app；
    待审队列指向每测试独立的临时文件并清空，避免互相污染。
    """
    monkeypatch.setenv("OFFLINE_DEMO_MODE", "1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_REVIEW_DECISION", raising=False)
    monkeypatch.setenv("REVIEW_STORE_PATH", str(tmp_path / "review_queue.json"))

    import audit.config as _cfg
    import audit.graph as _graph
    import audit.judge as _judge
    import audit.nodes as _nodes
    import audit.review_store as _store

    def _reset():
        _judge.reset_judge()
        _nodes.reset_llm()
        _graph.reset_app()
        _store.reset_store()   # 经 get_config() 读路径，可能顺带建 config 缓存……
        _cfg.reset_config()    # ……所以 config 放最后清，避免缓存到旧环境变量

    _reset()
    yield
    _reset()
