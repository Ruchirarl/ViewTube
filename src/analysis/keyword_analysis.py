"""
Keyword Analysis Module

Computes n-gram statistics from video text fields and measures lift on a target metric
(default: views_per_day).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Literal, Tuple, Optional, Set

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


TextSource = Literal["title", "description", "tags"]


class KeywordAnalyzer:
    """Compute n-gram lift against a numeric metric such as views_per_day."""

    def __init__(self) -> None:
        pass

    def compute_ngram_lift(
        self,
        df: pd.DataFrame,
        *,
        text_source: TextSource = "title",
        n: int = 2,
        min_support: int = 3,
        metric: str = "views_per_day",
        max_results: int = 50,
        stopwords: Optional[Iterable[str]] = None,
        min_token_length: int = 3,
        remove_numeric_tokens: bool = True,
    ) -> pd.DataFrame:
        """
        Compute n-gram lift: average metric for videos containing the n-gram
        divided by the overall average metric.

        Args:
            df: Processed videos dataframe
            text_source: Which text field to use (title, description, tags)
            n: n-gram size (1-3 typical)
            min_support: Minimum number of videos containing the n-gram
            metric: Target metric column (e.g., views_per_day)
            max_results: Max rows to return (after sorting by lift desc)

        Returns:
            DataFrame with columns: ngram, count, avg_metric, baseline_avg, lift
        """
        if metric not in df.columns:
            raise ValueError(f"Metric column '{metric}' not found in dataframe")

        # Baseline average for the chosen metric (avoid division by zero)
        metric_series = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        nonzero = metric_series[metric_series > 0]
        baseline_avg = (nonzero.mean() if len(nonzero) > 0 else metric_series.mean()) or 0.0
        if baseline_avg == 0:
            # If everything is zero, lift will be undefined; early return
            logger.warning("Baseline metric average is 0; cannot compute lift meaningfully.")
            return pd.DataFrame(columns=[
                "ngram", "count", "avg_metric", "baseline_avg", "lift"
            ])

        # Build tokens per row depending on text source
        tokens_per_row: List[List[str]] = []
        stopwords_set: Set[str] = set((s or "").lower() for s in (stopwords or []))

        if text_source == "title":
            source_col = "title_clean" if "title_clean" in df.columns else "title"
            for text in df[source_col].fillna(""):
                tokens_per_row.append(
                    self._tokenize_text(
                        text,
                        stopwords=stopwords_set,
                        min_token_length=min_token_length,
                        remove_numeric_tokens=remove_numeric_tokens,
                    )
                )
        elif text_source == "description":
            source_col = "description_clean" if "description_clean" in df.columns else "description"
            for text in df[source_col].fillna(""):
                tokens_per_row.append(
                    self._tokenize_text(
                        text,
                        stopwords=stopwords_set,
                        min_token_length=min_token_length,
                        remove_numeric_tokens=remove_numeric_tokens,
                    )
                )
        elif text_source == "tags":
            # Tags come as comma-separated; break into tokens (lowercased words)
            for tags_str in df.get("tags_string", pd.Series([""] * len(df))).fillna(""):
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                tokens = []
                for t in tags:
                    tokens.extend(
                        self._tokenize_text(
                            t,
                            stopwords=stopwords_set,
                            min_token_length=min_token_length,
                            remove_numeric_tokens=remove_numeric_tokens,
                        )
                    )
                tokens_per_row.append(tokens)
        else:
            raise ValueError(f"Unsupported text_source: {text_source}")

        # Build n-grams per row
        n = max(1, min(int(n), 5))
        row_ngrams: List[List[str]] = [sorted(set(self._generate_ngrams(tokens, n))) for tokens in tokens_per_row]

        # Aggregate counts and metric sums
        ngram_counts: Dict[str, int] = defaultdict(int)
        ngram_metric_sum: Dict[str, float] = defaultdict(float)

        for row_idx, ngrams in enumerate(row_ngrams):
            row_metric = float(metric_series.iat[row_idx])
            if not ngrams:
                continue
            for ng in ngrams:
                ngram_counts[ng] += 1
                ngram_metric_sum[ng] += row_metric

        # Build dataframe
        records: List[Tuple[str, int, float, float, float]] = []
        for ng, count in ngram_counts.items():
            if count < min_support:
                continue
            avg_metric = ngram_metric_sum[ng] / max(count, 1)
            lift = avg_metric / baseline_avg if baseline_avg > 0 else np.nan
            records.append((ng, count, avg_metric, baseline_avg, lift))

        result = pd.DataFrame(records, columns=[
            "ngram", "count", "avg_metric", "baseline_avg", "lift"
        ])

        if result.empty:
            return result

        result = result.sort_values(["lift", "count"], ascending=[False, False]).head(max_results)
        return result

    def _tokenize_text(
        self,
        text: str,
        *,
        stopwords: Optional[Set[str]] = None,
        min_token_length: int = 3,
        remove_numeric_tokens: bool = True,
    ) -> List[str]:
        text = (text or "").lower()
        # Keep alphanumerics and spaces; collapse whitespace
        raw = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
        raw = " ".join(raw.split())
        tokens = []
        for t in raw.split(" ") if raw else []:
            if not t:
                continue
            if remove_numeric_tokens and t.isnumeric():
                continue
            if len(t) < max(1, int(min_token_length)):
                continue
            if stopwords and t in stopwords:
                continue
            tokens.append(t)
        return tokens

    def _generate_ngrams(self, tokens: List[str], n: int) -> Iterable[str]:
        if n <= 1:
            for t in tokens:
                yield t
            return
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i : i + n])


