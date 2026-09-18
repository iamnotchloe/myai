"""Equal text on distinct source pages must retain independently citable chunks."""

from langchain_core.documents import Document


def test_index_build_preserves_same_text_on_distinct_pages(monkeypatch, tmp_path):
    from myai_rag import indexing

    (tmp_path / "11_示例公司_report.pdf").touch()
    pages = [Document(page_content="报告摘要。", metadata={"page": index}) for index in (0, 1)]
    class Loader:
        def __init__(self, path):
            pass
        def load(self):
            return pages
    monkeypatch.setattr(indexing, "PyPDFLoader", Loader)
    chunks = indexing.load_and_split_pdf(tmp_path)
    assert len(chunks) == 2
    assert [chunk.metadata["page"] for chunk in chunks] == [0, 1]
    assert chunks[0].page_content == chunks[1].page_content
