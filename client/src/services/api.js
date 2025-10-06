const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

const handleResponse = async (response) => {
  const contentType = response.headers.get("content-type");
  const isJson = contentType && contentType.includes("application/json");
  const payload = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const message = payload?.detail || payload?.message || response.statusText || "Request failed";
    throw new Error(message);
  }

  return payload;
};

export const uploadDocument = async (file, documentId) => {
  const formData = new FormData();
  formData.append("file", file);
  if (documentId) {
    formData.append("document_id", documentId);
  }

  const response = await fetch(`${API_BASE_URL}/upload/document`, {
    method: "POST",
    body: formData,
  });

  return handleResponse(response);
};

export const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload/image`, {
    method: "POST",
    body: formData,
  });

  return handleResponse(response);
};

export const uploadAudio = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload/audio`, {
    method: "POST",
    body: formData,
  });

  return handleResponse(response);
};

export const queryRag = async (payload) => {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return handleResponse(response);
};

export const listDocuments = async (docType) => {
  const url = new URL(`${API_BASE_URL}/documents`);
  if (docType) {
    url.searchParams.set("doc_type", docType);
  }

  const response = await fetch(url.toString());
  return handleResponse(response);
};

export const getStats = async () => {
  const response = await fetch(`${API_BASE_URL}/stats`);
  return handleResponse(response);
};
