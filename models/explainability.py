import pandas as pd

class Explainer:
    def __init__(self, movies_df):
        # index by lowercase title for fast lookup
        self.df = movies_df.copy()
        self.df['_lookup'] = self.df['title'].str.lower().str.strip()
        self.df = self.df.set_index('_lookup')

    def _get_row(self, title):
        key = str(title).lower().strip()
        if key in self.df.index:
            return self.df.loc[key]
        return None

    def explain(self, rec_title, source_title=None, svd_score=None, content_score=None):
        """
        Returns a human-readable string explaining why rec_title was recommended.
        source_title  : the movie the user typed (content-based signal)
        svd_score     : normalized svd score (collaborative signal)
        content_score : normalized content score (content-based signal)
        """
        reasons = []
        rec_row = self._get_row(rec_title)

        if rec_row is None:
            return "Recommended based on your preferences"

        # ── Collaborative signal ──────────────────────
        if svd_score is not None and float(svd_score) > 0.6:
            reasons.append("👥 Users like you loved this")

        # ── Content-based signals ─────────────────────
        if source_title is not None:
            src_row = self._get_row(source_title)

            if src_row is not None:

                # shared genres
                rec_genres = set(str(rec_row.get('genres', '')).split())
                src_genres = set(str(src_row.get('genres', '')).split())
                shared_genres = rec_genres & src_genres
                if shared_genres:
                    display = ', '.join(list(shared_genres)[:2]).title()
                    reasons.append(f"🎭 Matches genre: {display}")

                # shared actors
                rec_actors = set(str(rec_row.get('Actors', '')).split())
                src_actors = set(str(src_row.get('Actors', '')).split())
                shared_actors = rec_actors & src_actors
                if shared_actors:
                    # actors are stored lowercase no-space e.g. "robertdowneyjr"
                    # just show first one as-is; it's still meaningful
                    reasons.append(f"🎬 Features same actor")

                # same director
                rec_dir = str(rec_row.get('Director', '')).strip()
                src_dir = str(src_row.get('Director', '')).strip()
                if rec_dir and rec_dir == src_dir and rec_dir != 'nan':
                    reasons.append(f"🎥 Same director")

        # ── Sentiment signal ──────────────────────────
        sentiment = float(rec_row.get('sentiment_score', 0) or 0)
        if sentiment > 0.7:
            reasons.append("🌟 Highly praised by audiences")

        # ── Rating signal ─────────────────────────────
        avg = float(rec_row.get('ave_rating', 0) or 0)
        if avg >= 4.0:
            reasons.append(f"⭐ Top-rated ({round(avg, 1)}/5)")

        # ── Fallback ──────────────────────────────────
        if not reasons:
            reasons.append("✨ Similar style and themes")

        return "  ·  ".join(reasons)