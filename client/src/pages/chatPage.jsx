import React, { useState, useRef, useCallback, useEffect, Fragment, useMemo } from "react";
import {
  Send,
  User,
  Bot,
  MessageSquare,
  MessageSquarePlus,
  Trash2,
  Paperclip,
  File,
  Loader2,
  AlertCircle,
  CheckCircle2,
  FileText,
  Image as ImageIcon,
  Mic,
  Timer,
  BarChart3,
  ExternalLink,
  Copy
} from "lucide-react";
import {
  uploadDocument,
  uploadImage,
  uploadAudio,
  queryRag,
  listDocuments,
  getStats,
  deleteDocument,
  deleteAllDocuments,
  buildDocumentDownloadUrl
} from "../services/api";
const initialBotMessage = {
  sender: "bot",
  text: "Hello! 👋 I'm Quanta Quest, your multimodal AI copilot. Upload a file or ask a question to get started.",
};

const STORAGE_KEY = "ragSessions_v1";
const ACTIVE_SESSION_KEY = "ragSessions_active";

const cloneInitialBotMessage = () => ({ ...initialBotMessage });

const generateSessionId = () => {
  if (typeof window !== "undefined" && window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
};

const createSessionRecord = (title = "Chat 1") => ({
  id: generateSessionId(),
  title,
  messages: [cloneInitialBotMessage()],
  createdAt: Date.now(),
  updatedAt: Date.now(),
});

const sanitiseSessionRecord = (session, fallbackTitle) => ({
  id: session?.id || generateSessionId(),
  title: session?.title || fallbackTitle,
  messages:
    Array.isArray(session?.messages) && session.messages.length
      ? session.messages
      : [cloneInitialBotMessage()],
  createdAt: session?.createdAt || Date.now(),
  updatedAt: session?.updatedAt || Date.now(),
});

const loadSessionsFromStorage = () => {
  if (typeof window === "undefined") {
    return [createSessionRecord()];
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [createSessionRecord()];
    }

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || !parsed.length) {
      return [createSessionRecord()];
    }

    return parsed.map((session, index) => sanitiseSessionRecord(session, `Chat ${index + 1}`));
  } catch (error) {
    console.warn("Failed to load chat sessions from storage", error);
    return [createSessionRecord()];
  }
};

const getLargestSessionIndex = (sessions) => {
  if (!Array.isArray(sessions) || !sessions.length) {
    return 1;
  }

  return sessions.reduce((accumulator, session, index) => {
    const match = String(session?.title ?? "").match(/(\d+)(?!.*\d)/);
    const numericValue = match ? Number(match[1]) : index + 1;
    return Number.isFinite(numericValue) ? Math.max(accumulator, numericValue) : accumulator;
  }, sessions.length);
};

const deriveTitleFromPrompt = (prompt) => {
  if (!prompt) {
    return "New chat";
  }

  const trimmed = prompt.trim().replace(/\s+/g, " ");
  if (!trimmed) {
    return "New chat";
  }

  return trimmed.length > 42 ? `${trimmed.slice(0, 39)}…` : trimmed;
};

const getSessionPreview = (session) => {
  if (!session?.messages?.length) {
    return "No messages yet.";
  }

  const lastMessage = [...session.messages].reverse().find((message) => message?.text);
  if (!lastMessage?.text) {
    return "No messages yet.";
  }

  const sanitized = lastMessage.text.replace(/\s+/g, " ").trim();
  if (!sanitized) {
    return "No messages yet.";
  }

  return sanitized.length > 60 ? `${sanitized.slice(0, 57)}…` : sanitized;
};

const formatUpdatedTimestamp = (timestamp) => {
  if (!timestamp) {
    return "";
  }

  try {
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) {
      return "";
    }

    const now = new Date();
    const isSameDay = date.toDateString() === now.toDateString();
    if (isSameDay) {
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    return date.toLocaleDateString();
  } catch (error) {
    console.warn("Failed to format timestamp", error);
    return "";
  }
};

const AVAILABLE_MODELS = [
  {
    label: "Phi-3 Mini (fast)",
    value: "microsoft/phi-3-mini-4k-instruct",
    description: "Great for quick Q&A and factual answers."
  },
  {
    label: "Phi-3 Medium", 
    value: "microsoft/phi-3-medium-4k-instruct",
    description: "Balances depth and speed for longer responses."
  },
  {
    label: "Phi-3 Vision", 
    value: "microsoft/phi-3-vision",
    description: "Best when combining text with images."
  }
];

const TIMING_ORDER = [
  "analysis_ms",
  "retrieval_ms",
  "context_build_ms",
  "generation_ms",
  "total_ms"
];

const TIMING_LABELS = {
  analysis_ms: "Analysis",
  retrieval_ms: "Retrieval",
  context_build_ms: "Context",
  generation_ms: "Generation",
  total_ms: "Total"
};

const ChatPage = () => {
  const initialSessionsRef = useRef();
  if (!initialSessionsRef.current) {
    initialSessionsRef.current = loadSessionsFromStorage();
  }

  const [sessions, setSessions] = useState(initialSessionsRef.current);
  const [activeSessionId, setActiveSessionId] = useState(() => {
    if (typeof window !== "undefined") {
      const savedActiveId = window.localStorage.getItem(ACTIVE_SESSION_KEY);
      if (savedActiveId && initialSessionsRef.current.some((session) => session.id === savedActiveId)) {
        return savedActiveId;
      }
    }
    return initialSessionsRef.current[0]?.id ?? null;
  });
  const [input, setInput] = useState("");
  const [uploads, setUploads] = useState([]);
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState(null);
  const [errorBanner, setErrorBanner] = useState(null);
  const [deletingDocId, setDeletingDocId] = useState(null);
  const [isClearingCache, setIsClearingCache] = useState(false);
  const [selectedModel, setSelectedModel] = useState(AVAILABLE_MODELS[0].value);
  const [recentlyCopiedId, setRecentlyCopiedId] = useState(null);
  const sessionCounterRef = useRef(getLargestSessionIndex(initialSessionsRef.current));
  const fileInputRef = useRef(null);
  const currentSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) || sessions[0] || null,
    [sessions, activeSessionId]
  );
  const messages = currentSession?.messages ?? [];

  const refreshSystemData = useCallback(async () => {
    try {
      const [docsResponse, statsResponse] = await Promise.all([
        listDocuments().catch(() => null),
        getStats().catch(() => null),
      ]);

      if (docsResponse?.documents) {
        setDocuments(docsResponse.documents);
      }

      if (statsResponse) {
        setStats(statsResponse);
      }
    } catch (error) {
      console.error("Failed to refresh system data", error);
    }
  }, []);

  useEffect(() => {
    refreshSystemData();
  }, [refreshSystemData]);

  const updateSession = useCallback((sessionId, updater) => {
    if (!sessionId) {
      return;
    }

    setSessions((prev) =>
      prev.map((session) => {
        if (session.id !== sessionId) {
          return session;
        }

        const nextState = updater(session);
        if (!nextState) {
          return session;
        }

        const shouldRefreshTimestamp =
          Object.prototype.hasOwnProperty.call(nextState, "messages") ||
          Object.prototype.hasOwnProperty.call(nextState, "title") ||
          Object.prototype.hasOwnProperty.call(nextState, "updatedAt");

        const updatedAt = shouldRefreshTimestamp
          ? nextState.updatedAt !== undefined
            ? nextState.updatedAt
            : Date.now()
          : session.updatedAt;

        return {
          ...session,
          ...nextState,
          updatedAt,
        };
      })
    );
  }, []);

  const appendMessages = useCallback(
    (sessionId, newMessages) => {
      if (!sessionId || !Array.isArray(newMessages) || newMessages.length === 0) {
        return;
      }

      updateSession(sessionId, (session) => ({
        messages: [...(session.messages || []), ...newMessages],
      }));
    },
    [updateSession]
  );

  const handleNewChat = useCallback(() => {
    const nextIndex = (sessionCounterRef.current || 0) + 1;
    sessionCounterRef.current = nextIndex;
    const newSession = createSessionRecord(`Chat ${nextIndex}`);

    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setInput("");
    setErrorBanner(null);
    setIsSending(false);
  }, []);

  const handleClearAllChats = useCallback(() => {
    if (!sessions.length) {
      return;
    }

    if (typeof window !== "undefined") {
      const confirmation = window.confirm(
        "Clear all chats? This will remove every saved conversation and start a fresh session."
      );
      if (!confirmation) {
        return;
      }
    }

    const freshSession = createSessionRecord("Chat 1");
    initialSessionsRef.current = [freshSession];
    sessionCounterRef.current = 1;

    setSessions([freshSession]);
    setActiveSessionId(freshSession.id);
    setInput("");
    setErrorBanner(null);

    try {
      if (typeof window !== "undefined") {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify([freshSession]));
        window.localStorage.setItem(ACTIVE_SESSION_KEY, freshSession.id);
      }
    } catch (error) {
      console.warn("Unable to persist cleared chat state", error);
    }
  }, [sessions]);

  const handleSelectSession = useCallback(
    (sessionId) => {
      if (!sessionId || sessionId === activeSessionId) {
        return;
      }

      setActiveSessionId(sessionId);
      setInput("");
      setErrorBanner(null);
      setIsSending(false);
    },
    [activeSessionId]
  );

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    } catch (error) {
      console.warn("Unable to persist chat sessions", error);
    }
  }, [sessions]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      if (!activeSessionId) {
        window.localStorage.removeItem(ACTIVE_SESSION_KEY);
      } else {
        window.localStorage.setItem(ACTIVE_SESSION_KEY, activeSessionId);
      }
    } catch (error) {
      console.warn("Unable to persist active chat session", error);
    }
  }, [activeSessionId]);

  useEffect(() => {
    if ((!activeSessionId || !sessions.some((session) => session.id === activeSessionId)) && sessions[0]) {
      setActiveSessionId(sessions[0].id);
    }
  }, [sessions, activeSessionId]);

  useEffect(() => {
    sessionCounterRef.current = Math.max(
      sessionCounterRef.current || 0,
      getLargestSessionIndex(sessions)
    );
  }, [sessions]);

  useEffect(() => {
    if (!recentlyCopiedId) {
      return undefined;
    }

    const timeout = setTimeout(() => setRecentlyCopiedId(null), 2000);
    return () => clearTimeout(timeout);
  }, [recentlyCopiedId]);

  const resolveUploadTarget = (file) => {
    const mime = file.type?.toLowerCase() || "";
    const extension = file.name.split(".").pop()?.toLowerCase() || "";

    if (mime.startsWith("image/") || ["png", "jpg", "jpeg", "gif", "bmp", "webp"].includes(extension)) {
      return { handler: uploadImage, type: "image" };
    }

    if (mime.startsWith("audio/") || ["mp3", "wav", "m4a", "flac", "aac", "ogg"].includes(extension)) {
      return { handler: uploadAudio, type: "audio" };
    }

    if (
      mime === "application/pdf" ||
      mime === "application/msword" ||
      mime === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
      ["pdf", "doc", "docx", "txt", "md", "rtf"].includes(extension)
    ) {
      return { handler: uploadDocument, type: "document" };
    }

    return null;
  };

  const buildUploadSummary = (file, type, response) => {
    const parts = [`${file.name} uploaded successfully.`];

    if (type === "document" && response?.doc_id) {
      parts.push(`Document ID: ${response.doc_id}`);
      if (response?.num_chunks !== undefined) {
        parts.push(`Chunks indexed: ${response.num_chunks}`);
      }
    }

    if (type === "image" && response?.image_id) {
      parts.push(`Image ID: ${response.image_id}`);
      if (typeof response?.has_text === "boolean") {
        parts.push(response.has_text ? "Detected embedded text." : "No text detected in image.");
      }
    }

    if (type === "audio" && response?.audio_id) {
      parts.push(`Audio ID: ${response.audio_id}`);
      if (response?.duration) {
        parts.push(`Duration: ${response.duration.toFixed(1)}s`);
      }
    }

    const timing = response?.processing_time_ms ?? response?.processing_time;
    if (timing) {
      const value = typeof timing === "number" ? timing : Number(timing);
      if (!Number.isNaN(value)) {
        parts.push(`Processing time: ${Math.round(value)} ms`);
      }
    }

    return parts.join("\n");
  };

  const updateUploadStatus = (id, patch) => {
    setUploads((prev) => prev.map((upload) => (upload.id === id ? { ...upload, ...patch } : upload)));
  };

  const handleFileSelect = async (event) => {
    const selectedFiles = Array.from(event.target.files || []);
    if (!selectedFiles.length) {
      return;
    }

    setErrorBanner(null);
    setIsUploading(true);

    const sessionId = currentSession?.id;
    if (!sessionId) {
      setErrorBanner("No active chat session is available. Start a new chat first.");
      setIsUploading(false);
      return;
    }

    for (const file of selectedFiles) {
      const target = resolveUploadTarget(file);
      const uploadId = `${file.name}-${Date.now()}`;

      if (!target) {
        appendMessages(sessionId, [
          {
            sender: "bot",
            text: `⚠️ ${file.name} has an unsupported format. Please upload PDF, DOCX, image, or audio files.`,
            isError: true,
          },
        ]);
        continue;
      }

      setUploads((prev) => [
        ...prev,
        {
          id: uploadId,
          name: file.name,
          type: target.type,
          status: "uploading",
        },
      ]);

      try {
        const response = await target.handler(file);

        updateUploadStatus(uploadId, {
          status: "success",
          details: response,
        });

        appendMessages(sessionId, [
          {
            sender: "bot",
            text: buildUploadSummary(file, target.type, response),
            variant: "upload",
          },
        ]);
      } catch (error) {
        console.error("Upload failed", error);
        updateUploadStatus(uploadId, {
          status: "error",
          error: error.message,
        });

        appendMessages(sessionId, [
          {
            sender: "bot",
            text: `❌ Failed to upload ${file.name}: ${error.message}`,
            isError: true,
          },
        ]);
        setErrorBanner(error.message);
      }
    }

    setIsUploading(false);
    event.target.value = "";
    refreshSystemData();
  };

  const handleSend = async () => {
    if (!input.trim() || isSending) {
      return;
    }

    const question = input.trim();
    const userMessage = { sender: "user", text: question };
    const loadingMessage = {
      sender: "bot",
      text: "",
      isLoading: true,
      meta: null,
    };
    const sessionId = currentSession?.id;
    if (!sessionId) {
      return;
    }

    updateSession(sessionId, (session) => {
      const baseMessages = session.messages || [];
      const updatedMessages = [...baseMessages, userMessage, loadingMessage];
      const hasUserMessage = baseMessages.some((message) => message.sender === "user");

      return {
        messages: updatedMessages,
        ...(hasUserMessage ? {} : { title: deriveTitleFromPrompt(question) }),
      };
    });
    setInput("");
    setIsSending(true);
    setErrorBanner(null);

    try {
  const response = await queryRag({ query: question, top_k: 4 });

      updateSession(sessionId, (session) => {
        const updated = [...(session.messages || [])];
        const placeholderIndex = [...updated].reverse().findIndex((msg) => msg.isLoading);
        const actualIndex = placeholderIndex === -1 ? -1 : updated.length - 1 - placeholderIndex;

        const botMessage = {
          sender: "bot",
          text: response.answer,
          citations: response.citations,
          searchResults: response.search_results,
          meta: {
            processingTime: response.processing_time_ms,
            sources: response.num_sources,
            stageTimings: response.stage_timings || null,
          },
        };

        if (actualIndex >= 0) {
          updated[actualIndex] = botMessage;
        } else {
          updated.push(botMessage);
        }

        return {
          messages: updated,
        };
      });
    } catch (error) {
      console.error("Query failed", error);
      updateSession(sessionId, (session) => {
        const updated = [...(session.messages || [])];
        const placeholderIndex = [...updated].reverse().findIndex((msg) => msg.isLoading);
        const actualIndex = placeholderIndex === -1 ? -1 : updated.length - 1 - placeholderIndex;
        const errorMessage = {
          sender: "bot",
          text: `I ran into an issue while processing your question: ${error.message}`,
          isError: true,
        };

        if (actualIndex >= 0) {
          updated[actualIndex] = errorMessage;
        } else {
          updated.push(errorMessage);
        }

        return {
          messages: updated,
        };
      });
      setErrorBanner(error.message);
    } finally {
      setIsSending(false);
      refreshSystemData();
    }
  };

  const renderUploadIcon = (type) => {
    switch (type) {
      case "image":
        return <ImageIcon className="w-4 h-4 text-yellow-400" />;
      case "audio":
        return <Mic className="w-4 h-4 text-yellow-400" />;
      default:
        return <FileText className="w-4 h-4 text-yellow-400" />;
    }
  };

  const renderUploadStatus = (status) => {
    if (status === "success") {
      return <CheckCircle2 className="w-4 h-4 text-green-400" />;
    }
    if (status === "error") {
      return <AlertCircle className="w-4 h-4 text-red-400" />;
    }
    return <Loader2 className="w-4 h-4 text-yellow-400 animate-spin" />;
  };

  const getModalityIcon = (modality) => {
    switch ((modality || "text").toLowerCase()) {
      case "image":
        return <ImageIcon className="h-4 w-4 text-yellow-300" />;
      case "audio":
        return <Mic className="h-4 w-4 text-emerald-300" />;
      default:
        return <FileText className="h-4 w-4 text-sky-300" />;
    }
  };

  const formatLocationLabel = (location) => {
    if (!location || location === "N/A") {
      return "Referenced";
    }
    return location;
  };

  const handleOpenSource = (docId) => {
    const url = buildDocumentDownloadUrl(docId);
    if (!url) {
      setErrorBanner("Source file is no longer available for download.");
      return;
    }
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const handleCopyDocId = async (docId) => {
    if (!docId) {
      return;
    }

    try {
      await navigator.clipboard.writeText(docId);
      setRecentlyCopiedId(docId);
    } catch (error) {
      console.error("Failed to copy doc ID", error);
      setErrorBanner("Unable to copy document ID. You can copy it manually from the citation card.");
    }
  };

  const handleModelChange = (event) => {
    const { value } = event.target;
    setSelectedModel(value);
    const picked = AVAILABLE_MODELS.find((model) => model.value === value);
    const sessionId = currentSession?.id;
    if (sessionId) {
      appendMessages(sessionId, [
        {
          sender: "bot",
          text: `✅ Model preference set to ${picked?.label || value}.`
        }
      ]);
    }
  };

  const handleDeleteDocument = async (docId, docSource) => {
    if (!docId) {
      return;
    }
    setDeletingDocId(docId);
    setErrorBanner(null);
    const sessionId = currentSession?.id;

    try {
      const response = await deleteDocument(docId);
      if (sessionId) {
        appendMessages(sessionId, [
          {
            sender: "bot",
            text: `🧹 Removed ${docSource || docId} from the cache. ${response.message || ""}`.trim(),
          },
        ]);
      }
    } catch (error) {
      if (sessionId) {
        appendMessages(sessionId, [
          {
            sender: "bot",
            text: `❌ Failed to delete ${docSource || docId}: ${error.message}`,
            isError: true,
          },
        ]);
      }
      setErrorBanner(error.message);
    } finally {
      setDeletingDocId(null);
      refreshSystemData();
    }
  };

  const handleClearCache = async () => {
    setIsClearingCache(true);
    setErrorBanner(null);
    const sessionId = currentSession?.id;

    try {
      const response = await deleteAllDocuments();
      if (sessionId) {
        appendMessages(sessionId, [
          {
            sender: "bot",
            text: `🗑️ Cleared cached documents. ${response.message || ""}`.trim(),
          },
        ]);
      }
    } catch (error) {
      if (sessionId) {
        appendMessages(sessionId, [
          {
            sender: "bot",
            text: `❌ Failed to clear cache: ${error.message}`,
            isError: true,
          },
        ]);
      }
      setErrorBanner(error.message);
    } finally {
      setIsClearingCache(false);
      refreshSystemData();
    }
  };

  const renderRichText = (text, isUser = false) => {
    if (!text) {
      return null;
    }

    const paragraphs = text
      .split(/\n{2,}/)
      .map((block) => block.trim())
      .filter(Boolean);

    if (paragraphs.length === 0) {
      return (
        <p className={`text-sm leading-relaxed ${isUser ? "text-slate-900" : "text-gray-200"}`}>
          {text}
        </p>
      );
    }

    return paragraphs.map((paragraph, index) => {
      const lines = paragraph.split("\n");
      return (
        <p
          key={`${index}-${paragraph.slice(0, 20)}`}
          className={`text-sm leading-relaxed ${isUser ? "text-slate-900" : "text-gray-200"}`}
        >
          {lines.map((line, lineIdx) => (
            <Fragment key={`${index}-${lineIdx}`}>
              {line}
              {lineIdx < lines.length - 1 && <br />}
            </Fragment>
          ))}
        </p>
      );
    });
  };

  return (
  <main className="flex h-screen flex-col md:flex-row bg-gradient-to-br from-slate-950 via-gray-900 to-slate-950 text-white">
      {/* Sidebar */}
      <aside className="flex flex-col w-full md:w-64 bg-gray-800 border-t md:border-t-0 md:border-r border-gray-700 p-4 space-y-4 overflow-y-auto">
        <button
          onClick={handleNewChat}
          className="flex items-center gap-2 bg-yellow-400 text-black px-4 py-2 rounded-md font-semibold hover:bg-yellow-500 transition whitespace-nowrap"
        >
          <MessageSquarePlus className="w-4 h-4" /> New Chat
        </button>
        <button
          onClick={handleClearAllChats}
          disabled={sessions.length === 0}
          className="flex items-center gap-2 bg-gray-700/80 text-gray-200 px-4 py-2 rounded-md font-semibold hover:bg-gray-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Trash2 className="w-4 h-4" /> Clear All Chats
        </button>

        <div className="space-y-3">
          <h2 className="flex items-center gap-2 text-gray-300 font-semibold">
            <MessageSquare className="w-5 h-5 text-yellow-400" />
            Your Chats
          </h2>
          {sessions.length === 0 ? (
            <p className="text-gray-500 text-sm">Start a conversation to see it here.</p>
          ) : (
            <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
              {sessions.map((session) => {
                const isActive = session.id === currentSession?.id;
                return (
                  <button
                    key={session.id}
                    onClick={() => handleSelectSession(session.id)}
                    className={`w-full rounded-md border px-3 py-2 text-left transition ${
                      isActive
                        ? "border-yellow-400/80 bg-yellow-500/10 text-yellow-100 shadow-[0_12px_24px_-18px_rgba(250,204,21,0.65)]"
                        : "border-gray-700/60 bg-gray-700/50 text-gray-200 hover:border-yellow-400/40 hover:bg-gray-700/80"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 text-xs font-semibold">
                      <span className="truncate">{session.title}</span>
                      <span className="text-[10px] font-medium text-gray-400">
                        {formatUpdatedTimestamp(session.updatedAt)}
                      </span>
                    </div>
                    <p className="mt-1 text-[11px] text-gray-400 truncate">
                      {getSessionPreview(session)}
                    </p>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Model selector */}
        <div className="bg-gray-800/60 border border-gray-700 rounded-md p-3 space-y-2">
          <div className="flex items-center justify-between text-sm text-gray-300">
            <span className="font-semibold">Model preference</span>
          </div>
          <select
            value={selectedModel}
            onChange={handleModelChange}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-xs text-gray-200 focus:outline-none focus:ring-1 focus:ring-yellow-400"
          >
            {AVAILABLE_MODELS.map((model) => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
          <p className="text-[11px] text-gray-400">
            {AVAILABLE_MODELS.find((model) => model.value === selectedModel)?.description}
          </p>
        </div>

        {/* Upload status */}
        <div className="space-y-3 mt-4">
          <h2 className="flex items-center gap-2 text-gray-300 font-semibold">
            <File className="w-5 h-5 text-yellow-400" />
            Recent Uploads
          </h2>
          {uploads.length === 0 && (
            <p className="text-gray-500 text-sm">Upload a file to see progress here.</p>
          )}
          {uploads.slice().reverse().map((upload) => (
            <div
              key={upload.id}
              className="flex items-center justify-between bg-gray-700/70 p-2 rounded-md"
            >
              <div className="flex items-center gap-2 truncate max-w-[140px]">
                {renderUploadIcon(upload.type)}
                <span className="text-sm truncate">{upload.name}</span>
              </div>
              <div>{renderUploadStatus(upload.status)}</div>
            </div>
          ))}
        </div>

        {/* Indexed documents overview */}
        <div className="space-y-3 mt-6">
          <h2 className="flex items-center gap-2 text-gray-300 font-semibold">
            <FileText className="w-5 h-5 text-yellow-400" />
            Indexed Content
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={handleClearCache}
              disabled={isClearingCache || documents.length === 0}
              className="flex-1 flex items-center justify-center gap-2 text-xs bg-gray-700/70 px-2 py-1 rounded-md hover:bg-gray-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isClearingCache ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <span role="img" aria-label="broom">
                  🧹
                </span>
              )}
              Clear Cache
            </button>
          </div>
          {documents.length === 0 ? (
            <p className="text-gray-500 text-sm">No indexed items yet.</p>
          ) : (
            <div className="space-y-2">
              {documents.slice(-6).reverse().map((doc) => (
                <div key={doc.id} className="bg-gray-700/70 p-2 rounded-md text-xs space-y-1">
                  <div className="flex items-start justify-between gap-2">
                    <div className="truncate">
                      <p className="font-semibold text-gray-200 truncate">{doc.source}</p>
                      <p className="text-gray-400 capitalize">Type: {doc.doc_type}</p>
                    </div>
                    <button
                      onClick={() => handleDeleteDocument(doc.id, doc.source)}
                      disabled={deletingDocId === doc.id}
                      className="flex items-center gap-1 bg-gray-900/60 hover:bg-red-500/20 text-red-300 px-2 py-1 rounded-md transition disabled:opacity-60 disabled:cursor-not-allowed"
                    >
                      {deletingDocId === doc.id ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <span role="img" aria-label="trash">
                          🗑️
                        </span>
                      )}
                      Delete
                    </button>
                  </div>
                </div>
              ))}
              {documents.length > 6 && (
                <p className="text-[10px] text-gray-500">Showing latest 6 of {documents.length} documents.</p>
              )}
            </div>
          )}
        </div>

        {/* Stats */}
        {stats && (
          <div className="space-y-2 mt-6 text-xs bg-gray-700/50 p-3 rounded-md">
            <p className="text-sm font-semibold text-gray-200">System stats</p>
            <p className="text-gray-400">Documents: {stats.metadata_store?.total_documents ?? stats.total_documents ?? 0}</p>
            <p className="text-gray-400">Indexed: {stats.metadata_store?.indexed_documents ?? stats.indexed_documents ?? 0}</p>
            <p className="text-gray-400">Chunks: {stats.metadata_store?.total_chunks ?? stats.total_chunks ?? 0}</p>
            <p className="text-gray-400">Queries: {stats.metadata_store?.total_queries ?? stats.total_queries ?? 0}</p>
          </div>
        )}
      </aside>

      {/* Chat Section */}
      <section className="flex-1 flex flex-col">
        {errorBanner && (
          <div className="bg-red-500/10 text-red-300 text-sm px-4 py-2 border-b border-red-400/40">
            {errorBanner}
          </div>
        )}
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto px-3 py-4 md:px-8 md:py-8">
          <div className="mx-auto flex max-w-4xl flex-col space-y-6">
            {messages.map((msg, index) => {
              const isUser = msg.sender === "user";
              const isError = Boolean(msg.isError);
              const meta = msg.meta || {};
              const stageTimings = meta.stageTimings || null;

              return (
                <div
                  key={index}
                  className={`flex gap-3 ${isUser ? "flex-row-reverse text-right" : "text-left"}`}
                >
                  <div className="flex h-10 w-10 flex-none items-center justify-center rounded-full bg-gradient-to-br from-yellow-400/80 via-amber-400/90 to-yellow-500 text-gray-900 shadow-lg ring-1 ring-yellow-300/60">
                    {isUser ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
                  </div>

                  <article
                    className={`relative w-full max-w-2xl rounded-3xl border backdrop-blur transition-all ${
                      isUser
                        ? "bg-yellow-400 text-black border-yellow-300 shadow-[0_22px_46px_-28px_rgba(250,204,21,0.55)]"
                        : isError
                          ? "bg-red-500/15 border-red-400/40 text-red-100 shadow-[0_18px_35px_-28px_rgba(239,68,68,0.65)]"
                          : "bg-gray-900/70 border-gray-700/60 text-gray-100 shadow-[0_20px_40px_-32px_rgba(15,23,42,0.9)]"
                    } ${isUser ? "rounded-br-xl" : "rounded-bl-xl"}`}
                  >
                    <div className={`space-y-4 px-4 py-3 md:px-6 md:py-5 ${isUser ? "text-slate-900" : ""}`}>
                      <header
                        className={`flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-wide ${
                          isUser ? "justify-end text-black/70" : "justify-between text-gray-400"
                        }`}
                      >
                        <span>{isUser ? "You" : "Assistant"}</span>
                        {!isUser && meta?.processingTime !== undefined && (
                          <span className="inline-flex items-center gap-1 rounded-full bg-gray-900/60 px-2 py-1 text-[10px] font-medium text-gray-300">
                            <Timer className="h-3 w-3" />
                            {Math.round(meta.processingTime || 0)} ms
                          </span>
                        )}
                        {!isUser && typeof meta?.sources === "number" && (
                          <span className="inline-flex items-center gap-1 rounded-full bg-gray-900/60 px-2 py-1 text-[10px] font-medium text-gray-300">
                            <BarChart3 className="h-3 w-3" />
                            {meta.sources} {meta.sources === 1 ? "source" : "sources"}
                          </span>
                        )}
                      </header>

                      <div className="space-y-3 text-sm leading-relaxed">
                        {msg.isLoading ? (
                          <div className="flex items-center gap-3 text-gray-300">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            <span className="text-sm">Preparing your answer…</span>
                          </div>
                        ) : (
                          renderRichText(msg.text, isUser)
                        )}
                      </div>

                      {!isUser && stageTimings && (
                        <details className="group/stages mt-3 rounded-2xl border border-gray-700/60 bg-gray-900/50 px-4 py-3">
                          <summary className="flex cursor-pointer items-center justify-between text-xs font-semibold text-gray-300">
                            <span>Timing breakdown</span>
                            <span className="text-[11px] text-gray-500 group-open/stages:text-yellow-300">View details</span>
                          </summary>
                          <div className="mt-3 grid gap-2 sm:grid-cols-2">
                            {TIMING_ORDER.filter((key) => stageTimings[key] !== undefined).map((key) => (
                              <div
                                key={key}
                                className="rounded-xl border border-gray-700/50 bg-gray-900/60 px-3 py-2 text-left"
                              >
                                <p className="text-[11px] uppercase tracking-wide text-gray-400">
                                  {TIMING_LABELS[key] || key}
                                </p>
                                <p className="text-sm font-semibold text-gray-100">
                                  {Math.round(stageTimings[key])} ms
                                </p>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}

                      {msg.citations?.length > 0 && (
                        <div className="mt-4 space-y-3">
                          <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-400">
                            Citations
                          </h4>
                          <div className="grid gap-2">
                            {msg.citations.map((citation, citationIndex) => (
                              <div
                                key={`${citation.source_id || citationIndex}-${citationIndex}`}
                                className="rounded-xl border border-gray-700/50 bg-gray-900/60 px-3 py-3 text-left space-y-3"
                              >
                                <div className="flex flex-wrap items-start justify-between gap-3">
                                  <div className="flex items-start gap-2">
                                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gray-800/70">
                                      {getModalityIcon(citation.source_type)}
                                    </div>
                                    <div>
                                      <p className="text-sm font-semibold text-gray-200">
                                        {citation.source_name || "Unknown source"}
                                      </p>
                                      <p className="text-xs text-gray-400">
                                        {formatLocationLabel(citation.location)}
                                      </p>
                                    </div>
                                  </div>
                                  {citation.doc_id && (
                                    <button
                                      onClick={() => handleOpenSource(citation.doc_id)}
                                      className="inline-flex items-center gap-1 rounded-md border border-gray-700/60 bg-gray-900/70 px-2 py-1 text-xs font-semibold text-gray-200 hover:border-yellow-400/60 hover:text-yellow-300 transition"
                                    >
                                      <ExternalLink className="h-3 w-3" />
                                      View source
                                    </button>
                                  )}
                                </div>
                                {citation.text_snippet && (
                                  <p className="rounded-lg border-l-2 border-yellow-400/60 bg-gray-900/80 px-3 py-2 text-sm text-gray-200">
                                    {citation.text_snippet}
                                  </p>
                                )}
                                {citation.doc_id && (
                                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-gray-400">
                                    <span className="rounded-md bg-gray-900/70 px-2 py-1 font-mono text-xs text-gray-300">
                                      Doc ID: {citation.doc_id}
                                    </span>
                                    <button
                                      onClick={() => handleCopyDocId(citation.doc_id)}
                                      className="inline-flex items-center gap-1 rounded-md border border-gray-700/60 bg-gray-900/70 px-2 py-1 text-[11px] text-gray-200 hover:border-yellow-400/60 hover:text-yellow-300 transition"
                                    >
                                      <Copy className="h-3 w-3" />
                                      Copy ID
                                    </button>
                                    {recentlyCopiedId === citation.doc_id && (
                                      <span className="text-emerald-400 text-[11px]">Copied!</span>
                                    )}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {msg.searchResults?.length > 0 && (
                        <div className="mt-4 space-y-3">
                          <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-400">
                            Top matches
                          </h4>
                          <div className="grid gap-2">
                            {msg.searchResults.map((result) => (
                              <div
                                key={result.id}
                                className="rounded-xl border border-gray-700/50 bg-gray-900/60 px-3 py-3 text-left space-y-2"
                              >
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                  <div className="flex items-center gap-2">
                                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gray-800/70">
                                      {getModalityIcon(result.modality)}
                                    </div>
                                    <div>
                                      <p className="text-sm font-semibold text-gray-200">
                                        {result.metadata?.source ?? "Unknown source"}
                                      </p>
                                      <p className="text-xs text-gray-400 capitalize">
                                        {result.modality || result.metadata?.type || "text"}
                                      </p>
                                    </div>
                                  </div>
                                  {result.metadata?.doc_id && (
                                    <button
                                      onClick={() => handleOpenSource(result.metadata.doc_id)}
                                      className="inline-flex items-center gap-1 rounded-md border border-gray-700/60 bg-gray-900/70 px-2 py-1 text-xs font-semibold text-gray-200 hover:border-yellow-400/60 hover:text-yellow-300 transition"
                                    >
                                      <ExternalLink className="h-3 w-3" />
                                      View source
                                    </button>
                                  )}
                                </div>
                                {result.text && (
                                  <p className="rounded-lg border-l-2 border-gray-700/60 bg-gray-900/70 px-3 py-2 text-sm text-gray-300">
                                    {result.text}
                                  </p>
                                )}
                                {result.metadata?.doc_id && (
                                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-gray-400">
                                    <span className="rounded-md bg-gray-900/70 px-2 py-1 font-mono text-xs text-gray-300">
                                      Doc ID: {result.metadata.doc_id}
                                    </span>
                                    {recentlyCopiedId === result.metadata.doc_id ? (
                                      <span className="text-emerald-400 text-[11px]">Copied!</span>
                                    ) : (
                                      <button
                                        onClick={() => handleCopyDocId(result.metadata.doc_id)}
                                        className="inline-flex items-center gap-1 rounded-md border border-gray-700/60 bg-gray-900/70 px-2 py-1 text-[11px] text-gray-200 hover:border-yellow-400/60 hover:text-yellow-300 transition"
                                      >
                                        <Copy className="h-3 w-3" />
                                        Copy ID
                                      </button>
                                    )}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </article>
                </div>
              );
            })}
          </div>
        </div>

        {/* Input Box */}
        <div className="p-2 md:p-4 border-t border-gray-700 bg-gray-800 flex items-center space-x-2 md:space-x-3">
          <input
            type="text"
            className="flex-1 px-4 py-2 rounded-md bg-gray-900 border border-gray-600 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />

          {/* Hidden file input */}
          <input
            type="file"
            ref={fileInputRef}
            multiple
            accept=".png,.jpg,.jpeg,.gif,.bmp,.webp,.pdf,.doc,.docx,.txt,.md,.rtf,.mp3,.wav,.m4a,.flac,.aac,.ogg"
            className="hidden"
            onChange={handleFileSelect}
          />

          {/* Attachment Button */}
          <button
            onClick={() => fileInputRef.current.click()}
            className="p-2 bg-gray-700 text-white rounded-md hover:bg-gray-600 transition disabled:opacity-60 disabled:cursor-not-allowed"
            disabled={isUploading}
          >
            {isUploading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Paperclip className="w-5 h-5" />
            )}
          </button>

          {/* Send Button */}
          <button
            onClick={handleSend}
            className="p-2 bg-yellow-400 text-black rounded-md hover:bg-yellow-500 transition disabled:opacity-60 disabled:cursor-not-allowed"
            disabled={isSending}
          >
            {isSending ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
          </button>
        </div>
      </section>
    </main>
  );
};

export default ChatPage;
