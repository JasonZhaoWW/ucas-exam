from src.api.chunking import chunk_text


class TestChunkText:
    def test_empty_input_returns_empty_list(self):
        assert chunk_text("") == []

    def test_single_sentence_returns_single_chunk(self):
        assert chunk_text("这是一个句子。") == ["这是一个句子。"]

    def test_split_on_chinese_period(self):
        text = "第一句话。第二句话。"
        result = chunk_text(text)
        assert "".join(result) == text
        assert all(len(c) <= 500 for c in result)

    def test_split_on_all_chinese_punctuation_and_newline(self):
        text = "问题一？感叹一！句号一。换行一\n结束。"
        result = chunk_text(text)
        assert "".join(result) == text
        assert all(len(c) <= 500 for c in result)

    def test_merge_short_sentences_up_to_max(self):
        sentences = ["短句。" for _ in range(200)]
        text = "".join(sentences)
        chunks = chunk_text(text)
        assert len(chunks) < 200
        for chunk in chunks[:-1]:
            assert len(chunk) <= 500
        assert "".join(chunks) == text

    def test_long_single_sentence_stays_as_own_chunk(self):
        text = "这" * 600 + "。"
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_mixed_chinese_english_text(self):
        text = "Hello世界。This is a test！中文English混合？End."
        result = chunk_text(text)
        assert "".join(result) == text
        assert all(len(c) <= 500 for c in result)
