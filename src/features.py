"""
features.py
Defines the standard data structures for Candidate features to ensure
consistent extraction and simplified DataFrame conversion.
"""

from dataclasses import dataclass

@dataclass
class CandidateScore:
    candidate_id: str
    skill_score: float = 0.0
    domain_relevance_score: float = 0.0
    behavior_score: float = 0.0
    career_score: float = 0.0
    trust_score: float = 0.0
    is_honeypot: bool = False
    final_score: float = 0.0
    
    def to_dict(self):
        """Used to rapidly serialize to dictionaries for Pandas ingestion."""
        return {
            "candidate_id": self.candidate_id,
            "skill_score": self.skill_score,
            "domain_relevance_score": self.domain_relevance_score,
            "behavior_score": self.behavior_score,
            "career_score": self.career_score,
            "trust_score": self.trust_score,
            "is_honeypot": self.is_honeypot,
            "final_score": self.final_score
        }
