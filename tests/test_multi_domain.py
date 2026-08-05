import unittest
from equimind.domain_adapter.schema import DomainType, DomainQueryContext, DomainResearchResult
from equimind.domain_adapter.legal_adapter import LegalResearchAdapter
from equimind.domain_adapter.medical_adapter import MedicalReviewAdapter
from equimind.domain_adapter.cybersecurity_adapter import CybersecurityThreatAdapter


class TestMultiDomainAdapters(unittest.TestCase):

    def test_legal_domain_adapter(self):
        ctx = DomainQueryContext(
            domain=DomainType.LEGAL_CASE_RESEARCH,
            entity_name="United States v. TechCorp",
            user_query="Analyze antitrust precedents for TechCorp merger",
        )
        res = LegalResearchAdapter.execute_legal_research(ctx)
        self.assertIsInstance(res, DomainResearchResult)
        self.assertEqual(res.domain, DomainType.LEGAL_CASE_RESEARCH)
        self.assertGreater(res.confidence_score, 0.8)
        self.assertTrue(len(res.citations) >= 1)

    def test_medical_domain_adapter(self):
        ctx = DomainQueryContext(
            domain=DomainType.HEALTHCARE_MEDICAL_REVIEW,
            entity_name="OncoDrug-X",
            user_query="Review Phase III clinical trial efficacy for OncoDrug-X",
        )
        res = MedicalReviewAdapter.execute_medical_review(ctx)
        self.assertIsInstance(res, DomainResearchResult)
        self.assertEqual(res.domain, DomainType.HEALTHCARE_MEDICAL_REVIEW)
        self.assertIn("Clinical Trial", res.summary_verdict)

    def test_cybersecurity_domain_adapter(self):
        ctx = DomainQueryContext(
            domain=DomainType.CYBERSECURITY_THREAT_INTEL,
            entity_name="CVE-2024-9999",
            user_query="Assess CVSS score and exploit probability for CVE-2024-9999",
        )
        res = CybersecurityThreatAdapter.execute_threat_intel(ctx)
        self.assertIsInstance(res, DomainResearchResult)
        self.assertEqual(res.domain, DomainType.CYBERSECURITY_THREAT_INTEL)
        self.assertIn("CVSS", res.summary_verdict)


if __name__ == "__main__":
    unittest.main()
