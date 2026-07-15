import { useEffect, useState } from "react";
import {
  ArrowRight,
  BookMarked,
  BookOpen,
  Library,
  Star,
  BookHeart
} from "lucide-react";

import "./App.css";
import {
  getRecommendations,
  getUserProfile,
  getUsers,
} from "./api/api";

function formatScore(score) {
  if (typeof score !== "number" || Number.isNaN(score)) {
    return "N/A";
  }
  return score.toFixed(1);
}

function buildReasons(book) {
  const reasons = [];

  if (book.genre && book.genre !== "General") {
    reasons.push(`Perfect for fans of ${book.genre}`);
  }

  if (typeof book.rating === "number" && book.rating > 4.0) {
    reasons.push(`Highly rated with a ${book.rating.toFixed(1)} star average`);
  }

  if (typeof book.knn_score === "number" && book.knn_score > 0.5) {
    reasons.push("Loved by readers with similar taste");
  }

  if (typeof book.cb_score === "number" && book.cb_score > 0.5) {
    reasons.push("Matches topics you frequently read");
  }

  if (reasons.length === 0) {
    reasons.push("A classic favorite we think you'll enjoy");
  }

  return reasons.slice(0, 3);
}

function formatDate(value) {
  if (!value) return "Unknown date";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-US", {
    month: "long",
    year: "numeric",
  });
}

function formatTransactionType(value) {
  if (!value) return "Read";
  if (value.includes("purchase")) return "Purchased";
  if (value.includes("rental")) return "Borrowed";
  return "Read";
}

export default function App() {
  const [screen, setScreen] = useState("home");
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState("");
  const [userProfile, setUserProfile] = useState(null);
  const [isProfileLoading, setIsProfileLoading] = useState(false);

  useEffect(() => {
    async function loadUsers() {
      try {
        const data = await getUsers();
        setUsers(data);
        setSelectedUser(data[0] ?? "");
      } catch (err) {
        console.error(err);
        setError("Couldn't connect to the digital library.");
      }
    }
    loadUsers();
  }, []);

  useEffect(() => {
    if (!selectedUser) return;
    let cancelled = false;

    async function loadUserProfile() {
      setIsProfileLoading(true);
      try {
        const profile = await getUserProfile(selectedUser);
        if (!cancelled) {
          setUserProfile(profile);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setUserProfile(null);
        }
      } finally {
        if (!cancelled) setIsProfileLoading(false);
      }
    }
    loadUserProfile();
    return () => { cancelled = true; };
  }, [selectedUser]);

  async function generateRecommendations() {
    if (!selectedUser) {
      setError("Please select a reader profile first.");
      return;
    }

    setError("");
    setScreen("loading");

    try {
      const books = await getRecommendations(selectedUser, 10);
      setRecommendations(books);
      setTimeout(() => {
        setScreen("recommendations");
      }, 800);
    } catch (err) {
      console.error(err);
      setError("We couldn't fetch recommendations right now.");
      setScreen("dashboard");
    }
  }

  return (
    <div className="app">
      <nav className="navbar">
        <div className="logo" onClick={() => setScreen("home")} style={{cursor: "pointer"}}>
          <BookHeart size={28} />
          <span>Pustaka</span>
        </div>

        <div className="navLinks">
          <button onClick={() => setScreen("home")}>Home</button>
          <button onClick={() => setScreen("dashboard")}>My Library</button>
          <button
            className="secondary"
            style={{ padding: "8px 16px", borderRadius: "8px", fontSize: "14px", marginLeft: "12px" }}
            onClick={() => window.open("https://github.com/", "_blank")}
          >
            GitHub
          </button>
        </div>
      </nav>

      {screen === "home" && (
        <section className="hero">
          <div className="badge">
            <BookOpen size={15} />
            Your Personal Digital Library
          </div>

          <h1>
            Discover your next
            <br />
            great read.
          </h1>

          <p>
            Welcome to Pustaka. Our smart curation system learns your unique tastes 
            to bring you hand-picked recommendations you're guaranteed to love.
          </p>

          <div className="heroButtons">
            <button className="primary" onClick={() => setScreen("dashboard")}>
              Enter Library
              <ArrowRight size={18} />
            </button>
          </div>

          <div className="stats">
            <div className="statCard">
              <h2>1,200+</h2>
              <span>Curated Books</span>
            </div>
            <div className="statCard">
              <h2>50,000+</h2>
              <span>Happy Readers</span>
            </div>
            <div className="statCard">
              <h2>4.8</h2>
              <span>Average Rating</span>
            </div>
            <div className="statCard">
              <h2>99%</h2>
              <span>Match Accuracy</span>
            </div>
          </div>
        </section>
      )}

      {screen === "dashboard" && (
        <section className="dashboard">
          <div className="profile">
            <h1>{userProfile?.user_name ?? "Reader"}</h1>
            <p>Member since 2021</p>

            <div className="history">
              <h3>Recently Read</h3>
              {isProfileLoading ? (
                <div>Loading your shelf...</div>
              ) : (
                userProfile?.recent_books?.slice(0, 4).map((book) => (
                  <div key={book.transaction_id}>
                    <strong>{book.title}</strong>
                    {book.author}
                    <br />
                    {formatTransactionType(book.transaction_type)} in {formatDate(book.transaction_date)}
                  </div>
                ))
              )}
            </div>

            <div className="genres">
              <h3>Favorite Genres</h3>
              {!isProfileLoading && userProfile?.favorite_genres?.map((genre) => (
                <span key={genre}>{genre}</span>
              ))}
            </div>
          </div>

          <div className="engine">
            <h2>Personalized Curation</h2>
            <p>
              Select your reader profile below to view your personalized book feed. 
              Our system blends your past reading history with similar readers to find 
              hidden gems just for you.
            </p>

            <select
              value={selectedUser}
              onChange={(event) => setSelectedUser(event.target.value)}
            >
              <option value="" disabled>Select a Reader Profile</option>
              {users.map((user) => (
                <option key={user} value={user}>
                  Reader Profile: {user}
                </option>
              ))}
            </select>

            {error && <p style={{ color: "#d97757", marginTop: "-10px" }}>{error}</p>}

            <button
              className="primary"
              onClick={generateRecommendations}
              disabled={!selectedUser}
              style={{ width: "100%", justifyContent: "center" }}
            >
              Curate My Reading List
            </button>
          </div>
        </section>
      )}

      {screen === "loading" && (
        <section className="loading">
          <div className="loader-spinner" />
          <h1>Curating your shelf...</h1>
          <p>Browsing through thousands of titles to find your perfect match.</p>
        </section>
      )}

      {screen === "recommendations" && (
        <section className="recommendations">
          <h1>Recommended For You</h1>
          <p className="subtitle">
            Hand-picked titles based on your unique taste profile.
          </p>

          <div className="recommendationGrid">
            {recommendations.map((book) => {
              const reasons = buildReasons(book);

              return (
                <div key={book.book_id} className="recommendationCard">
                  {typeof book.score === 'number' && (
                    <div className="matchBadge">
                      {(book.score * 100).toFixed(0)}% Match
                    </div>
                  )}
                  
                  <div className="cover">
                    <Library size={48} />
                  </div>

                  <h2>{book.title}</h2>
                  <p>{book.author}</p>
                  
                  <div className="score">
                    <Star size={16} fill="currentColor" />
                    {formatScore(book.rating)} / 5.0
                  </div>

                  <div className="reasonBox">
                    <h4>Why this book?</h4>
                    {reasons.map((reason, i) => (
                      <p key={i}>{reason}</p>
                    ))}
                    
                    <div className="analyticsBar">
                      <span>CB: <strong>{typeof book.cb_score === 'number' ? book.cb_score.toFixed(2) : 'N/A'}</strong></span>
                      <span>KNN: <strong>{typeof book.knn_score === 'number' ? book.knn_score.toFixed(2) : 'N/A'}</strong></span>
                      <span>SVD: <strong>{typeof book.svd_score === 'number' ? book.svd_score.toFixed(2) : 'N/A'}</strong></span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ display: "flex", justifyContent: "center", marginTop: "60px" }}>
            <button className="secondary" onClick={() => setScreen("dashboard")}>
              Back to Dashboard
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
