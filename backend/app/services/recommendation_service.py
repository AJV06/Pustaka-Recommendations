import pandas as pd
from pathlib import Path


class RecommendationService:

    def __init__(self):

        root = Path(__file__).resolve().parents[3]

        self.hybrid = pd.read_csv(
            root / "models/hybrid/hybrid_recommendations.csv"
        )

        self.books = pd.read_csv(
            root / "data/cleaned/books_clean.csv"
        )

        self.users = pd.read_csv(
            root / "data/cleaned/users_clean.csv"
        )

        self.transactions = pd.read_csv(
            root / "data/cleaned/transactions_clean.csv"
        )

    def _clean_records(self, frame, columns=None):

        if columns is not None:
            frame = frame[columns]

        cleaned = frame.copy().astype(object).where(pd.notna(frame), None)

        return cleaned.to_dict(orient="records")

    def _clean_scalar(self, value):

        if pd.isna(value):
            return None

        return value

    def recommend(self, user_id, top_n=10):

        recs = (
            self.hybrid[
                self.hybrid["user_id"] == user_id
            ]
            .sort_values("rank")
            .head(top_n)
        )

        result = recs.merge(
            self.books,
            on="book_id",
            how="left"
        )

        return self._clean_records(result)

    def get_user_profile(self, user_id, history_limit=4, genre_limit=4):

        user_rows = self.users[self.users["user_id"] == user_id]

        if user_rows.empty:
            return None

        user = user_rows.iloc[0]

        history = (
            self.transactions[self.transactions["user_id"] == user_id]
            .sort_values("transaction_date", ascending=False)
            .head(history_limit)
            .merge(
                self.books[
                    [
                        "book_id",
                        "title",
                        "author",
                        "genre",
                        "rating",
                    ]
                ],
                on="book_id",
                how="left",
            )
        )

        favorite_genres = (
            self.transactions[self.transactions["user_id"] == user_id]
            .merge(
                self.books[["book_id", "genre"]],
                on="book_id",
                how="left",
            )["genre"]
            .dropna()
            .value_counts()
            .head(genre_limit)
            .index
            .tolist()
        )

        recent_books = self._clean_records(
            history,
            [
                "transaction_id",
                "transaction_date",
                "transaction_type",
                "book_id",
                "title",
                "author",
                "genre",
                "rating",
            ],
        )

        return {
            "user_id": self._clean_scalar(user["user_id"]),
            "user_name": self._clean_scalar(user["user_name"]),
            "preferred_language": self._clean_scalar(user["preferred_language"]),
            "city": self._clean_scalar(user["city"]),
            "state": self._clean_scalar(user["state"]),
            "subscription_type": self._clean_scalar(user["subscription_type"]),
            "recent_books": recent_books,
            "favorite_genres": favorite_genres,
        }
