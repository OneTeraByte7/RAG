import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Send,
  User,
  Bot,
  Plus,
  Paperclip,
  File,
  Loader2,
  AlertCircle,
  CheckCircle2,
  FileText,
  Image as ImageIcon,
  Mic,
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
} from "../services/api";
const initialBotMessage = {
  sender: "bot",
  text: "Hello! 👋 I'm your AI assistant. Upload a file or ask a question to get started.",
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

const ChatPage = () => {
  const [messages, setMessages] = useState([initialBotMessage]);
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
  const fileInputRef = useRef(null);

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

    for (const file of selectedFiles) {
      const target = resolveUploadTarget(file);
      const uploadId = `${file.name}-${Date.now()}`;

      if (!target) {
        setMessages((prev) => [
          ...prev,
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

        setMessages((prev) => [
          ...prev,
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

        setMessages((prev) => [
          ...prev,
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
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsSending(true);
    setErrorBanner(null);

    try {
      const response = await queryRag({ query: question, top_k: 6 });

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: response.answer,
          citations: response.citations,
          searchResults: response.search_results,
          meta: {
            processingTime: response.processing_time_ms,
            sources: response.num_sources,
          },
        },
      ]);
    } catch (error) {
      console.error("Query failed", error);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `I ran into an issue while processing your question: ${error.message}`,
          isError: true,
        },
      ]);
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

  const handleModelChange = (event) => {
    const { value } = event.target;
    setSelectedModel(value);
    const picked = AVAILABLE_MODELS.find((model) => model.value === value);
    setMessages((prev) => [
      ...prev,
      {
        sender: "bot",
        text: `✅ Model preference set to ${picked?.label || value}. (UI-only setting)`
      }
    ]);
  };

  const handleDeleteDocument = async (docId, docSource) => {
    if (!docId) {
      return;
    }
    setDeletingDocId(docId);
    setErrorBanner(null);

    try {
      const response = await deleteDocument(docId);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `🧹 Removed ${docSource || docId} from the cache. ${response.message || ""}`.trim(),
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `❌ Failed to delete ${docSource || docId}: ${error.message}`,
          isError: true,
        },
      ]);
      setErrorBanner(error.message);
    } finally {
      setDeletingDocId(null);
      refreshSystemData();
    }
  };

  const handleClearCache = async () => {
    setIsClearingCache(true);
    setErrorBanner(null);

    try {
      const response = await deleteAllDocuments();
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `🗑️ Cleared cached documents. ${response.message || ""}`.trim(),
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `❌ Failed to clear cache: ${error.message}`,
          isError: true,
        },
      ]);
      setErrorBanner(error.message);
    } finally {
      setIsClearingCache(false);
      refreshSystemData();
    }
  };

  return (
    <main className="flex flex-col md:flex-row h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <aside className="flex flex-col w-full md:w-64 bg-gray-800 border-t md:border-t-0 md:border-r border-gray-700 p-4 space-y-4 overflow-y-auto">
        <button className="flex items-center gap-2 bg-yellow-400 text-black px-4 py-2 rounded-md font-semibold hover:bg-yellow-500 transition whitespace-nowrap">
          <Plus className="w-4 h-4" /> New Chat
        </button>

        {/* Model selector */}
        <div className="bg-gray-800/60 border border-gray-700 rounded-md p-3 space-y-2">
          <div className="flex items-center justify-between text-sm text-gray-300">
            <span className="font-semibold">Model preference</span>
            <span className="text-[10px] text-gray-500">UI only</span>
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
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex items-start space-x-3 ${msg.sender === "user" ? "justify-end" : "justify-start"
                }`}
            >
              {msg.sender === "bot" && (
                <Bot className="w-8 h-8 text-yellow-400" />
              )}
              <div
                className={`max-w-xs md:max-w-2xl px-4 py-3 rounded-2xl whitespace-pre-wrap break-words ${msg.sender === "user"
                    ? "bg-yellow-400 text-black rounded-br-none"
                    : msg.isError
                      ? "bg-red-500/10 text-red-200 border border-red-400/60 rounded-bl-none"
                      : "bg-gray-800 text-white rounded-bl-none"
                  }`}
              >
                <p>{msg.text}</p>
                {msg.meta && (
                  <p className="mt-2 text-xs text-gray-300">
                    Answered in {Math.round(msg.meta.processingTime || 0)} ms · Sources used: {msg.meta.sources ?? 0}
                  </p>
                )}
                {msg.citations?.length > 0 && (
                  <div className="mt-3 space-y-2 text-xs text-gray-300">
                    <p className="font-semibold text-gray-200">Citations</p>
                    {msg.citations.map((citation, index) => (
                      <div key={`${citation.source_id || index}-${index}`} className="bg-gray-900/40 p-2 rounded-md">
                        <p className="text-gray-200">{citation.source_name} · {citation.location}</p>
                        <p className="mt-1 text-gray-400">{citation.text_snippet}</p>
                      </div>
                    ))}
                  </div>
                )}
                {msg.searchResults?.length > 0 && (
                  <div className="mt-3 space-y-2 text-xs text-gray-300">
                    <p className="font-semibold text-gray-200">Top matches</p>
                    {msg.searchResults.map((result) => (
                      <div key={result.id} className="bg-gray-900/40 p-2 rounded-md">
                        <p className="text-gray-200">{result.metadata?.source ?? "Unknown source"}</p>
                        <p className="mt-1 text-gray-400">{result.text}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {msg.sender === "user" && (
                <User className="w-8 h-8 text-yellow-400" />
              )}
            </div>
          ))}
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
