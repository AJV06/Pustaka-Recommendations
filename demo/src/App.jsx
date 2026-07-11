import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  BookMarked,
  BookOpen,
  BrainCircuit,
  Library,
  Sparkles,
  Star,
  Users,
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

function getConfidence(book) {
  const score = Number(book.hybrid_score ?? 0);

  return Math.max(0, Math.min(100, Math.round(score * 100)));
}

function buildReasons(book) {
  const reasons = [];

  if (book.genre) {
    reasons.push(`Strong fit for ${book.genre} readers`);
  }

  if (typeof book.rating === "number") {
    reasons.push(`Well-rated title with a ${book.rating.toFixed(1)} star average`);
  }

  if (typeof book.knn_score === "number" && book.knn_score > 0) {
    reasons.push("Recommended from similar readers in the community");
  }

  if (typeof book.cb_score === "number" && book.cb_score > 0) {
    reasons.push("Matched through content-based similarity");
  }

  if (typeof book.svd_score === "number" && book.svd_score > 0) {
    reasons.push("Backed by latent preference patterns from SVD");
  }

  return reasons.slice(0, 3);
}

function formatDate(value) {
  if (!value) {
    return "Unknown date";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function formatTransactionType(value) {
  if (!value) {
    return "Activity";
  }

  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export default function App() {
  const [screen, setScreen] = useState("home");
  const [progress, setProgress] = useState(0);
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState("");
  const [userProfile, setUserProfile] = useState(null);
  const [profileError, setProfileError] = useState("");
  const [isProfileLoading, setIsProfileLoading] = useState(false);

  useEffect(() => {
    async function loadUsers() {
      try {
        const data = await getUsers();

        setUsers(data);
        setSelectedUser(data[0] ?? "");
      } catch (err) {
        console.error(err);
        setError("Couldn't load users from the recommendation API.");
      }
    }

    loadUsers();
  }, []);

  useEffect(() => {
    if (!selectedUser) {
      return;
    }

    let cancelled = false;

    async function loadUserProfile() {
      setIsProfileLoading(true);
      setProfileError("");

      try {
        const profile = await getUserProfile(selectedUser);

        if (!cancelled) {
          setUserProfile(profile);
        }
      } catch (err) {
        console.error(err);

        if (!cancelled) {
          setUserProfile(null);
          setProfileError("Couldn't load recent reading activity for this user.");
        }
      } finally {
        if (!cancelled) {
          setIsProfileLoading(false);
        }
      }
    }

    loadUserProfile();

    return () => {
      cancelled = true;
    };
  }, [selectedUser]);

  async function generateRecommendations() {
    if (!selectedUser) {
      setError("Select a user before generating recommendations.");
      return;
    }

    setError("");
    setProgress(10);
    setScreen("loading");

    const interval = window.setInterval(() => {
      setProgress((prev) => (prev >= 90 ? prev : prev + 5));
    }, 120);

    try {
      const books = await getRecommendations(selectedUser, 10);

      setProgress(100);
      setRecommendations(books);

      window.setTimeout(() => {
        setScreen("recommendations");
        setProgress(0);
      }, 500);
    } catch (err) {
      console.error(err);
      setError("Couldn't generate recommendations.");
      setScreen("dashboard");
      setProgress(0);
    } finally {
      window.clearInterval(interval);
    }
  }

  return (
    <div className="app">
      <div className="background" />

      <nav className="navbar">
        <div className="logo">
          <BookOpen />
          <span>Pustaka AI</span>
        </div>

        <div className="navLinks">
          <button onClick={() => setScreen("home")}>Home</button>
          <button onClick={() => setScreen("dashboard")}>Dashboard</button>
          <button
            className="githubButton"
            onClick={() =>
              window.open(
                "https://github.com/",
                "_blank",
                "noopener,noreferrer"
              )
            }
            aria-label="Open GitHub"
            title="Open GitHub"
          >
            <BookOpen size={18} />
          </button>
        </div>
      </nav>

      {screen === "home" && (
        <motion.section
          className="hero"
          initial={{ opacity: 0, y: 60 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="badge">
            <Sparkles size={15} />
            AI Powered Recommendation Platform
          </div>

          <h1>
            Discover your next
            <br />
            <span>favorite book.</span>
          </h1>

          <p>
            Personalized recommendations using
            <strong> Hybrid Machine Learning</strong>
            <br />
            SVD • KNN • Content Based Filtering
          </p>

          <div className="heroButtons">
            <button className="primary" onClick={() => setScreen("dashboard")}>
              Explore Platform
              <ArrowRight size={18} />
            </button>

            <button
              className="secondary"
              onClick={() =>
                window.open(
                  "https://fastapi.tiangolo.com/",
                  "_blank",
                  "noopener,noreferrer"
                )
              }
            >
              View Documentation
            </button>
          </div>

          <div className="stats">
            <div className="statCard">
              <Users size={38} />
              <h2>1000+</h2>
              <span>Users</span>
            </div>

            <div className="statCard">
              <Library size={38} />
              <h2>600+</h2>
              <span>Books</span>
            </div>

            <div className="statCard">
              <BrainCircuit size={38} />
              <h2>Hybrid</h2>
              <span>Recommendation Engine</span>
            </div>

            <div className="statCard">
              <BarChart3 size={38} />
              <h2>3728</h2>
              <span>Transactions</span>
            </div>
          </div>
        </motion.section>
      )}

      {screen === "dashboard" && (
        <motion.section
          className="dashboard"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="profile">
            <h1>Hello {userProfile?.user_name ?? selectedUser ?? "Reader"} 👋</h1>
            <p>Current User: {selectedUser || "Loading..."}</p>
            {userProfile && (
              <p>
                {userProfile.city}, {userProfile.state} •{" "}
                {userProfile.preferred_language} • {userProfile.subscription_type}
              </p>
            )}

            <div className="history">
              <h3>Recently Read</h3>
              {isProfileLoading && <div>Loading recent reading activity...</div>}
              {!isProfileLoading && profileError && <div>{profileError}</div>}
              {!isProfileLoading &&
                !profileError &&
                userProfile?.recent_books?.map((book) => (
                  <div key={book.transaction_id}>
                    <strong>{book.title}</strong>
                    <br />
                    {book.author} • {book.genre}
                    <br />
                    {formatTransactionType(book.transaction_type)} on{" "}
                    {formatDate(book.transaction_date)}
                  </div>
                ))}
              {!isProfileLoading &&
                !profileError &&
                userProfile?.recent_books?.length === 0 && (
                  <div>No recent reading activity found for this user.</div>
                )}
            </div>

            <div className="genres">
              <h3>Favourite Genres</h3>
              {isProfileLoading && <span>Loading...</span>}
              {!isProfileLoading &&
                userProfile?.favorite_genres?.map((genre) => (
                  <span key={genre}>{genre}</span>
                ))}
              {!isProfileLoading &&
                !profileError &&
                userProfile?.favorite_genres?.length === 0 && (
                  <span>No genre history yet</span>
                )}
            </div>
          </div>

          <div className="engine">
            <h2>Hybrid Recommendation Engine</h2>

            <p>
              Our recommendation engine combines Collaborative Filtering,
              Content Based Filtering, KNN and SVD Matrix Factorization to
              generate highly personalized book recommendations.
            </p>

            <div className="engineCards">
              <div className="miniCard">
                <BrainCircuit />
                Hybrid
              </div>

              <div className="miniCard">
                <Library />
                Content
              </div>

              <div className="miniCard">
                <Users />
                KNN
              </div>

              <div className="miniCard">
                <BarChart3 />
                SVD
              </div>
            </div>

            <select
              value={selectedUser}
              onChange={(event) => setSelectedUser(event.target.value)}
              style={{
                width: "100%",
                padding: "12px",
                marginTop: "20px",
                marginBottom: "20px",
                borderRadius: "10px",
                fontSize: "16px",
              }}
            >
              {users.map((user) => (
                <option key={user} value={user}>
                  {user}
                </option>
              ))}
            </select>

            {error && <p>{error}</p>}

            <button
              className="primary big"
              onClick={generateRecommendations}
              disabled={!selectedUser}
            >
              Generate Recommendations
            </button>
          </div>
        </motion.section>
      )}

      {screen === "loading" && (
        <motion.section
          className="loading"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <BrainCircuit size={90} />
          <h1>Generating Recommendations...</h1>
          <p>Initializing Hybrid Recommendation Engine</p>

          <div className="loader">
            <div className="loaderFill" style={{ width: `${progress}%` }} />
          </div>

          <span>{progress}%</span>

          <div className="steps">
            {progress < 25 && <span>Loading user profile...</span>}
            {progress >= 25 && progress < 50 && (
              <span>Finding similar readers...</span>
            )}
            {progress >= 50 && progress < 75 && (
              <span>Running Hybrid Model...</span>
            )}
            {progress >= 75 && <span>Ranking recommendations...</span>}
          </div>
        </motion.section>
      )}

      {screen === "recommendations" && (
        <motion.section
          className="recommendations"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <h1>Your Recommendations</h1>

          <p
            className="subtitle"
            style={{
              textAlign: "center",
              marginTop: "-35px",
              marginBottom: "40px",
              color: "#b8bfdc",
            }}
          >
            Generated using our Hybrid Recommendation Engine
          </p>

          <div className="recommendationGrid">
            {recommendations.map((book) => {
              const confidence = getConfidence(book);
              const reasons = buildReasons(book);

              return (
                <motion.div
                  key={book.book_id}
                  className="recommendationCard"
                  whileHover={{ y: -10, scale: 1.03 }}
                >
                  <div
                    className="cover"
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      background:
                        "linear-gradient(135deg, rgba(37,99,235,.22), rgba(124,58,237,.22))",
                      border: "1px solid rgba(255,255,255,.08)",
                    }}
                  >
                    <BookMarked size={80} />
                  </div>

                  <h2>{book.title}</h2>
                  <p>{book.author}</p>
                  <p>{book.genre}</p>

                  <div className="score">
                    <Star size={16} />
                    {formatScore(book.rating)}
                  </div>

                  <div className="reasonBox">
                    <h4>Why this book?</h4>
                    {reasons.length > 0 ? (
                      reasons.map((reason) => <p key={reason}>✓ {reason}</p>)
                    ) : (
                      <p>✓ Ranked highly by the hybrid recommendation engine</p>
                    )}
                  </div>

                  <div style={{ marginTop: "18px" }}>
                    <h4 style={{ marginBottom: "10px" }}>Match Confidence</h4>
                    <div
                      style={{
                        width: "100%",
                        height: "10px",
                        borderRadius: "999px",
                        overflow: "hidden",
                        background: "rgba(255,255,255,.08)",
                      }}
                    >
                      <div
                        style={{
                          width: `${confidence}%`,
                          height: "100%",
                          background:
                            "linear-gradient(90deg, #2563eb, #7c3aed)",
                        }}
                      />
                    </div>
                    <p style={{ marginTop: "12px", marginBottom: 0 }}>
                      <strong>{confidence}% Match</strong>
                    </p>
                  </div>
                </motion.div>
              );
            })}
          </div>

          <div
            style={{
              display: "flex",
              justifyContent: "center",
              marginTop: "60px",
              gap: "20px",
            }}
          >
            <button
              className="secondary"
              onClick={() => setScreen("dashboard")}
            >
              Back
            </button>

            <button
              className="primary"
              onClick={() => setScreen("analytics")}
            >
              View Analytics
            </button>
          </div>
        </motion.section>
      )}

      {screen === "analytics" && (
        <motion.section
          className="recommendations"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <h1>Model Performance</h1>

          <div className="stats">
            <div className="statCard">
              <BarChart3 />
              <h2>0.03</h2>
              <span>RMSE</span>
            </div>

            <div className="statCard">
              <Users />
              <h2>84%</h2>
              <span>Precision</span>
            </div>

            <div className="statCard">
              <Library />
              <h2>81%</h2>
              <span>Recall</span>
            </div>

            <div className="statCard">
              <BrainCircuit />
              <h2>Hybrid</h2>
              <span>Best Performing Model</span>
            </div>
          </div>

          <div
            style={{
              marginTop: "60px",
              display: "flex",
              justifyContent: "center",
            }}
          >
            <button className="primary" onClick={() => setScreen("home")}>
              Back to Home
            </button>
          </div>
        </motion.section>
      )}
    </div>
  );
}
