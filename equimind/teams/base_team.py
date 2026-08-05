from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional

from equimind.evidence.schema import EvidenceNode
from equimind.providers.base import LLMProvider


class ResearchTeam(ABC):
    """Abstract Base Class for specialized Research Subagent Teams."""

    @property
    @abstractmethod
    def team_name(self) -> str:
        """Returns canonical name of the research team."""
        pass

    @abstractmethod
    def research(
        self,
        ticker: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        provider: Optional[LLMProvider] = None,
        as_of_date: Optional[datetime] = None,
    ) -> List[EvidenceNode]:
        """Executes domain research and returns structured EvidenceNode list."""
        pass
