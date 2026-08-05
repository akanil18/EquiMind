import hashlib
import math
import re
from datetime import datetime, timezone
from typing import List, Set, Dict, Optional, Tuple

from equimind.evidence.schema import EvidenceNode, AuthorCredibility


CREDIBILITY_WEIGHTS: Dict[AuthorCredibility, float] = {
    AuthorCredibility.VERIFIED_OFFICIAL: 1.5,
    AuthorCredibility.HIGH: 1.2,
    AuthorCredibility.MEDIUM: 1.0,
    AuthorCredibility.LOW: 0.6,
}


class ContextCompressor:
    """In-memory deterministic compression engine for evidence nodes.
    
    Performs exact deduplication, fuzzy clustering, time-decay scoring,
    query relevance ranking, and token budget packing without requiring extra LLM calls.
    """

    @classmethod
    def compress(
        cls,
        nodes: List[EvidenceNode],
        query_context: str,
        max_token_budget: int = 4000,
        similarity_threshold: float = 0.7,
        as_of_date: Optional[datetime] = None,
    ) -> List[EvidenceNode]:
        """Runs full deterministic compression pipeline."""
        if not nodes:
            return []

        # 1. Temporal cutoff filtering (if running in backtesting mode)
        if as_of_date:
            nodes = [n for n in nodes if n.publication_timestamp <= as_of_date]

        # 2. Exact deduplication
        deduped = cls.exact_deduplicate(nodes)

        # 3. Fuzzy similarity clustering deduplication
        clustered = cls.fuzzy_cluster_deduplicate(deduped, similarity_threshold=similarity_threshold)

        # 4. Relevance ranking & time-decay scoring
        ranked = cls.score_and_rank(clustered, query_context, current_time=as_of_date)

        # 5. Pack nodes into context token budget
        packed = cls.pack_context_budget(ranked, max_token_budget=max_token_budget)

        return packed

    @classmethod
    def exact_deduplicate(cls, nodes: List[EvidenceNode]) -> List[EvidenceNode]:
        """Removes exact duplicate content nodes using MD5 content hash."""
        seen_hashes: Set[str] = set()
        unique_nodes: List[EvidenceNode] = []

        for node in nodes:
            norm_content = re.sub(r"\s+", " ", node.content.strip().lower())
            content_hash = hashlib.md5(norm_content.encode("utf-8")).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_nodes.append(node)

        return unique_nodes

    @classmethod
    def fuzzy_cluster_deduplicate(
        cls, nodes: List[EvidenceNode], similarity_threshold: float = 0.7
    ) -> List[EvidenceNode]:
        """Clusters semantically similar evidence (e.g. repeated news headlines) and retains the highest credibility node."""
        if len(nodes) <= 1:
            return nodes

        clusters: List[List[EvidenceNode]] = []

        for node in nodes:
            node_tokens = set(re.findall(r"\w+", node.content.lower()))
            if not node_tokens:
                clusters.append([node])
                continue

            matched_cluster = None
            for cluster in clusters:
                rep_tokens = set(re.findall(r"\w+", cluster[0].content.lower()))
                jaccard = cls._jaccard_similarity(node_tokens, rep_tokens)
                if jaccard >= similarity_threshold:
                    matched_cluster = cluster
                    break

            if matched_cluster:
                matched_cluster.append(node)
            else:
                clusters.append([node])

        # Pick highest scoring node from each cluster
        retained: List[EvidenceNode] = []
        for cluster in clusters:
            best_node = max(
                cluster,
                key=lambda n: CREDIBILITY_WEIGHTS.get(n.author_credibility, 1.0) * n.confidence_score,
            )
            retained.append(best_node)

        return retained

    @classmethod
    def score_and_rank(
        cls,
        nodes: List[EvidenceNode],
        query_context: str,
        current_time: Optional[datetime] = None,
    ) -> List[EvidenceNode]:
        """Scores and ranks evidence nodes by combining credibility, confidence, time decay, and relevance."""
        ref_time = current_time or datetime.now(timezone.utc)
        query_tokens = set(re.findall(r"\w+", query_context.lower()))

        scored_nodes: List[Tuple[float, EvidenceNode]] = []
        for node in nodes:
            cred_weight = CREDIBILITY_WEIGHTS.get(node.author_credibility, 1.0)
            conf_score = node.confidence_score

            # Time decay calculation (half life ~ 14 days)
            time_diff = (ref_time - node.publication_timestamp).total_seconds()
            days_old = max(0.0, time_diff / 86400.0)
            time_decay = math.exp(-0.05 * days_old)

            # Keyword relevance overlap
            node_tokens = set(re.findall(r"\w+", (node.title + " " + node.content).lower()))
            if query_tokens and node_tokens:
                overlap = len(query_tokens.intersection(node_tokens))
                relevance = 1.0 + (overlap / len(query_tokens))
            else:
                relevance = 1.0

            final_score = cred_weight * conf_score * time_decay * relevance
            scored_nodes.append((final_score, node))

        # Sort descending by score
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored_nodes]

    @classmethod
    def pack_context_budget(
        cls, nodes: List[EvidenceNode], max_token_budget: int = 4000
    ) -> List[EvidenceNode]:
        """Packs highest ranked nodes into token budget (approximating 1 token = 4 chars)."""
        current_tokens = 0
        packed: List[EvidenceNode] = []

        for node in nodes:
            # Estimate tokens for formatted node
            node_str = f"Source: {node.source_type.value} | Title: {node.title} | Content: {node.content}"
            est_tokens = len(node_str) // 4 + 10

            if current_tokens + est_tokens <= max_token_budget:
                packed.append(node)
                current_tokens += est_tokens
            else:
                # Stop if budget is reached
                break

        return packed

    @staticmethod
    def _jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
