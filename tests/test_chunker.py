"""
Tests for FinancialChunker and Parent-Child hierarchy.
"""

import unittest
from equimind.evidence.schema import EvidenceSource
from equimind.rag.chunker import FinancialChunker, FinancialChunk


class TestFinancialChunker(unittest.TestCase):

    def test_chunk_sec_filing(self):
        sec_text = (
            "Item 1. Business\nNVIDIA is a leader in accelerated computing and visual computing.\n\n"
            "Item 1A. Risk Factors\nWe face competition from cloud providers designing internal ASICs.\n\n"
            "Item 7. Management Discussion\nRevenue grew significantly across all segments."
        )
        chunks = FinancialChunker.chunk_document(
            content=sec_text,
            ticker="NVDA",
            source_type=EvidenceSource.SEC_FILING,
            max_chunk_chars=300,
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(any("Risk Factors" in (c.section_name or "") for c in chunks))

    def test_chunk_earnings_transcript(self):
        transcript_text = (
            "Jensen Huang: Blackwell demand is extraordinary and we are scaling production.\n\n"
            "Toshiya Hari - Goldman Sachs: Can you discuss the gross margins for next quarter?\n\n"
            "Colette Kress: We expect gross margins to remain around 75% for the full year."
        )
        chunks = FinancialChunker.chunk_document(
            content=transcript_text,
            ticker="NVDA",
            source_type=EvidenceSource.EARNINGS_TRANSCRIPT,
            max_chunk_chars=200,
        )
        self.assertGreater(len(chunks), 1)

    def test_parent_child_hierarchy(self):
        content = "\n\n".join(["Paragraph of financial data and market insights for technology sector analysis." for _ in range(20)])
        parents, children = FinancialChunker.create_parent_child_chunks(
            content=content,
            ticker="AAPL",
            source_type=EvidenceSource.FINANCIAL_NEWS,
            parent_chunk_chars=400,
            child_chunk_chars=120,
        )
        self.assertGreater(len(parents), 0)
        self.assertGreater(len(children), len(parents))
        for child in children:
            self.assertIsNotNone(child.parent_id)


if __name__ == "__main__":
    unittest.main()
