"""
Multi-Domain Orchestration Adaptability Suite for EquiMind.
"""

from .schema import DomainType, DomainQueryContext, DomainResearchResult
from .legal_adapter import LegalResearchAdapter
from .medical_adapter import MedicalReviewAdapter
from .cybersecurity_adapter import CybersecurityThreatAdapter

__all__ = [
    "DomainType",
    "DomainQueryContext",
    "DomainResearchResult",
    "LegalResearchAdapter",
    "MedicalReviewAdapter",
    "CybersecurityThreatAdapter",
]
