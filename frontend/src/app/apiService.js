const API_BASE_URL = 
  import.meta.env.VITE_API_BASE_URL || "https://yousynopsis.onrender.com";

function getToken() {
  return localStorage.getItem("token");
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = Object.prototype.hasOwnProperty.call(options, "token")
    ? options.token
    : getToken();

  if (!headers.has("Content-Type") && options.body && !(options.body instanceof URLSearchParams)) {
    headers.set("Content-Type", "application/json");
  }

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  let data = null;
  const text = await response.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const message = response.status === 401
      ? "Your session expired. Please sign in again."
      : data?.detail || data?.message || "Request failed";
    const error = new Error(message);
    error.status = response.status;
    if (response.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
    }
    throw error;
  }

  return data;
}

export const api = {
  baseUrl: API_BASE_URL,

  async register(payload) {
    return request("/api/auth/register", {
      method: "POST",
      body: JSON.stringify(payload),
      token: null,
    });
  },

  async login(email, password) {
    const formData = new URLSearchParams();
    formData.append("username", email);
    formData.append("password", password);

    return request("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData,
      token: null,
    });
  },

  async me() {
    return request("/api/users/me");
  },

  async updateProfile(payload) {
    return request("/api/users/profile", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  },

  async adminUsers() {
    return request("/api/admin/users");
  },

  async adminUsage() {
    return request("/api/admin/usage");
  },

  async summarize(youtubeUrl, mode = "normal", customPrompt = "", outputLanguage = "English") {
    return request("/api/summarize", {
      method: "POST",
      body: JSON.stringify({
        youtube_url: youtubeUrl,
        mode,
        custom_prompt: customPrompt || undefined,
        output_language: outputLanguage,
      }),
    });
  },

  async compareVideos(youtubeUrl1, youtubeUrl2, comparisonGoal = "", outputLanguage = "English") {
    return request("/api/compare-videos", {
      method: "POST",
      body: JSON.stringify({
        youtube_url_1: youtubeUrl1,
        youtube_url_2: youtubeUrl2,
        comparison_goal: comparisonGoal || undefined,
        output_language: outputLanguage,
      }),
    });
  },

  async videoFeatures(youtubeUrl, transcript) {
    return request("/api/video/features", {
      method: "POST",
      body: JSON.stringify({
        youtube_url: youtubeUrl,
        transcript,
      }),
    });
  },

  async hydrateSummary(summaryId) {
    return request(`/api/summaries/${summaryId}/hydrate`, {
      method: "POST",
    });
  },

  async recentSummaries() {
    return request("/api/summaries/recent");
  },

  async recentComparisons() {
    return request("/api/comparisons/recent");
  },

  async savePresentation(payload) {
    return request("/api/presentations", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async improveSlide(slide, context) {
    return request("/api/presentations/improve-slide", {
      method: "POST",
      body: JSON.stringify({ slide, context }),
    });
  },

  async askSummaryAi(question, summaryData, selectedWindow) {
    return request("/api/summary/chat", {
      method: "POST",
      body: JSON.stringify({
        question,
        summary: summaryData?.summary || "",
        transcript: summaryData?.transcript || "",
        caption_summaries: summaryData?.caption_summaries || [],
        selected_window: selectedWindow || undefined,
      }),
    });
  },

  async translateSummary(language, data) {
    return request("/api/summary/translate", {
      method: "POST",
      body: JSON.stringify({ language, data }),
    });
  },
};
