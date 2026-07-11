const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export async function getUsers() {
  const res = await fetch(`${API}/users`);

  if (!res.ok) {
    throw new Error(`Failed to load users: ${res.status}`);
  }

  return res.json();
}

export async function getRecommendations(userId, topN = 10) {
  const res = await fetch(`${API}/recommend`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: userId,
      top_n: topN,
    }),
  });

  if (!res.ok) {
    throw new Error(`Failed to load recommendations: ${res.status}`);
  }

  return res.json();
}

export async function getUserProfile(userId) {
  const res = await fetch(`${API}/users/${userId}/profile`);

  if (!res.ok) {
    throw new Error(`Failed to load user profile: ${res.status}`);
  }

  return res.json();
}
