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
    """In-memory deterministic compression and reranking engine for evidence nodes.
    
    Performs exact deduplication, fuzzy clustering, time-decay scoring,
    MMR (Maximal Marginal Relevance) diversity selection, and token budget packing.
    """

    @classmethod
    def compress(
        cls,
        nodes: List[EvidenceNode],
        query_context: str,
        max_token_budget: int = 4000,
        similarity_threshold: float = 0.7,
        as_of_date: Optional[datetime] = None,
        use_mmr: bool = True,
        mmr_lambda: float = 0.65,
    ) -> List[EvidenceNode]:
        """Runs full deterministic compression and diversity pipeline."""
        if not nodes:
            return []

        # 1. Temporal cutoff filtering
        if as_of_date:
            nodes = [n for n in nodes if n.publication_timestamp <= as_of_date]

        # 2. Exact deduplication
        deduped = cls.exact_deduplicate(nodes)

        # 3. Fuzzy similarity clustering deduplication
        clustered = cls.fuzzy_cluster_deduplicate(deduped, similarity_threshold=similarity_threshold)

        # 4. Relevance ranking & time-decay scoring
        ranked = cls.score_and_rank(clustered, query_context, current_time=as_of_date)

        # 5. Maximal Marginal Relevance (MMR) for diversity
        if use_mmr and len(ranked) > 2:
            diverse_ranked = cls.maximal_marginal_relevance(ranked, query_context, lambda_param=mmr_lambda)
        else:
            diverse_ranked = ranked

        # 6. Pack nodes into context token budget
        packed = cls.pack_context_budget(diverse_ranked, max_token_budget=max_token_budget)

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
        """Clusters semantically similar evidence and retains the highest credibility node."""
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

            time_diff = (ref_time - node.publication_timestamp).total_seconds()
            days_old = max(0.0, time_diff / 86400.0)
            time_decay = math.exp(-0.05 * days_old)

            node_tokens = set(re.findall(r"\w+", (node.title + " " + node.content).lower()))
            if query_tokens and node_tokens:
                overlap = len(query_tokens.intersection(node_tokens))
                relevance = 1.0 + (overlap / len(query_tokens))
            else:
                relevance = 1.0

            # Bonus from Agentic RAG retrieval score if present
            rag_bonus = (getattr(node, "rag_retrieval_score", None) or 0.5)

            final_score = cred_weight * conf_score * time_decay * relevance * (1.0 + 0.3 * rag_bonus)
            scored_nodes.append((final_score, node))

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored_nodes]

    @classmethod
    def maximal_marginal_relevance(
        cls,
        ranked_nodes: List[EvidenceNode],
        query_context: str,
        lambda_param: float = 0.65,
        top_k: int = 20,
    ) -> List[EvidenceNode]:
        """Applies MMR to balance query relevance with result diversity."""
        if not ranked_nodes:
            return []

        selected: List[EvidenceNode] = [ranked_nodes[0]]
        candidates = ranked_nodes[1:]
        
        node_token_sets = {
            n.id: set(re.findall(r"\w+", (n.title + " " + n.content).lower()))
            for n in ranked_nodes
        }
        query_tokens = set(re.findall(r"\w+", query_context.lower()))

        while candidates and len(selected) < top_k:
            best_score = -float("inf")
            best_candidate = None

            for cand in candidates:
                cand_tokens = node_token_sets[cand.id]
                
                # Sim 1: Query relevance
                rel = cls._jaccard_similarity(query_tokens, cand_tokens) if query_tokens else 0.5
                
                # Sim 2: Max similarity to already selected nodes
                max_sim_selected = max(
                    cls._jaccard_similarity(cand_tokens, node_token_sets[sel.id])
                    for sel in selected
                )

                mmr_score = (lambda_param * rel) - ((1.0 - lambda_param) * max_sim_selected)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = cand

            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break

        return selected

    @classmethod
    def pack_context_budget(
        cls, nodes: List[EvidenceNode], max_token_budget: int = 4000
    ) -> List[EvidenceNode]:
        """Packs highest ranked nodes into token budget."""
        current_tokens = 0
        packed: List[EvidenceNode] = []

        for node in nodes:
            node_str = f"Source: {node.source_type.value} | Title: {node.title} | Content: {node.content}"
            est_tokens = len(node_str) // 4 + 10

            if current_tokens + est_tokens <= max_token_budget:
                packed.append(node)
                current_tokens += est_tokens
            else:
                break

        return packed

    @staticmethod
    def _jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
