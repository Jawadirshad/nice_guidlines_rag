import React, { useState } from "react";
import { MessageCircle, Send, Plus, Clock, BookOpen, ExternalLink, LinkIcon } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import ContextModal from "@/helper/contextModel";

type Message = {
  role: "user" | "assistant";
  content: string;
  context?: string | null;
  references?: { link: string }[];
  timing?: {
    retrieval_time?: number;
    generation_time?: number;
    total_time?: number;
  };
  timestamp: Date;
};

type RAGResponse = {
  answer: string;
  context: string | null;
  results: any[];
  category: string;
  timing: {
    retrieval_time: number;
    generation_time: number;
    total_time: number;
  };
};

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
  ;

  const formatContent = (content: string) => {
    // Split content into lines, process each line, then join back
    const lines = content.split('\n');
    const processedLines = lines.map(line => {
      let processedLine = line;

      // Handle headers first
      if (line.trim().startsWith('###')) {
        return line.replace(
          /^###\s+(.+)$/,
          '<h3 class="text-xl font-semibold mt-4 mb-2">$1</h3>'
        );
      }
      if (line.trim().startsWith('##')) {
        return line.replace(
          /^##\s+(.+)$/,
          '<h2 class="text-2xl font-semibold mt-4 mb-2">$1</h2>'
        );
      }
      if (line.trim().startsWith('#')) {
        return line.replace(
          /^#\s+(.+)$/,
          '<h1 class="text-3xl font-bold mt-4 mb-3">$1</h1>'
        );
      }

      // Handle bullet points
      if (line.trim().startsWith('*') || line.trim().startsWith('-')) {
        return line.replace(
          /^\s*[\*\-]\s+(.+)$/,
          '<li class="ml-4 mb-2">$1</li>'
        );
      }

      // Handle links in the line
      processedLine = processedLine.replace(
        /\[([^\]]+)\]\(([^)]+)\)/g,
        (match, text, url) => {
          if (url.includes('nice.org.uk') || url.startsWith('www.nice.org.uk')) {
            url = `https://${url.replace(/^(?:https?:\/\/)?(?:www\.)?/, '')}`;
          }
          return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="text-blue-500 hover:underline">${text}</a>`;
        }
      );

      // Handle bold text
      processedLine = processedLine.replace(
        /\*\*([^*]+)\*\*/g,
        '<strong>$1</strong>'
      );

      return processedLine;
    });

    let formatted = processedLines.join('\n');

    // Wrap consecutive list items in ul
    formatted = formatted.replace(
      /(<li[^>]*>.*<\/li>\n?)+/g,
      '<ul class="list-disc mb-4">$&</ul>'
    );

    // Wrap regular text in p tags
    formatted = formatted.replace(
      /^(?!<[h|u|p|l])[^<\n].*$/gm,
      '<p class="mb-2">$&</p>'
    );

    return formatted;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { 
      role: "user", 
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prevMessages => {
      const updatedMessages = [...prevMessages, userMessage];
      return updatedMessages.slice(-16);
    });
    
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: userMessage.content,
          conversation_history: messages.map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: RAGResponse = await response.json();

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        context: data.context,
        references: data.results?.map(res => {
          const url = res.pdf_link || "";
          return {
            link: url.includes('nice.org.uk') || url.startsWith('www.nice.org.uk')
              ? `https://${url.replace(/^(?:https?:\/\/)?(?:www\.)?/, '')}`
              : url
          };
        }),
        timing: data.timing,
        timestamp: new Date()
      };

      setMessages(prevMessages => {
        const updatedMessages = [...prevMessages, assistantMessage];
        return updatedMessages.slice(-16);
      });
    } catch (error) {
      console.error("API call failed:", error);
      const errorMessage: Message = {
        role: "assistant",
        content: "I apologize, but I'm having trouble processing your request. Please try again.",
        timestamp: new Date()
      };

      setMessages(prevMessages => {
        const updatedMessages = [...prevMessages, errorMessage];
        return updatedMessages.slice(-16);
      });
    } finally {
      setIsLoading(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
  };

  const formatTime = (seconds: number) => {
    return `${seconds.toFixed(2)}s`;
  };

  return (
    <div className="flex h-screen flex-col bg-gradient-to-br from-cyan-50 to-green-50">
      <header className="border-b bg-white/60 backdrop-blur-lg">
        <div className="flex h-20 items-center px-6 gap-6">
          <div className="flex flex-1 items-center gap-4">
            <MessageCircle className="h-8 w-8 text-cyan-600" />
            <h2 className="text-3xl font-bold bg-gradient-to-r from-cyan-600 to-green-600 bg-clip-text text-transparent">
              NICE Guidelines Assistant
            </h2>
          </div>
          <button
            onClick={startNewChat}
            className="p-2 rounded-lg hover:bg-cyan-100 transition-colors"
          >
            <Plus className="h-5 w-5 text-cyan-600" />
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex flex-1 flex-col">
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                 className={`max-w-[80%] rounded-2xl p-6 shadow-lg ${
                  message.role === "user"
                    ? "bg-gradient-to-r from-cyan-500 to-green-500 text-white ml-16" // Changed text-white to text-black
                    : "bg-white text-black mr-16" // Added text-black for assistant
                }`}
                >
                  <div className="mb-2 text-sm opacity-70">
                    {message.role === "user" ? "You" : "Assistant"}
                  </div>

                  {/* Main answer content */}
                  <ReactMarkdown>{message.content}</ReactMarkdown>

                  {/* Context section */}
                  {/* {message.role === "assistant" && message.context && (
                    <div className="mt-6 pt-4 border-t border-gray-200">
                      <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                        <BookOpen className="h-5 w-5" />
                        Retrieved Context:
                      </h3>
                      <div className="bg-gray-50/50 rounded-lg p-4">
                        <pre className="whitespace-pre-wrap text-sm">
                          {message.context}
                        </pre>
                      </div>
                    </div>
                  )} */}

          {message.role === "assistant" && message.context && (
                    <ContextModal context={message.context} />
                  )}
                  

                  {/* References section */}
                  {message.role === "assistant" && message.references && message.references.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                        <LinkIcon className="h-5 w-5" />
                        References:
                      </h3>
                      <ul className="space-y-2">
                        {message.references.map((ref, idx) => (
                          <li key={idx}>
                            <a
                              href={ref.link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-500 hover:underline break-all"
                            >
                              {ref.link.replace(/^(?:https?:\/\/)?(?:www\.)?/, '')}
                            </a>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Timing information */}
                  {message.role === "assistant" && message.timing && (
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                        <Clock className="h-5 w-5" />
                        Processing Time:
                      </h3>
                      <div className="space-y-1 text-sm">
                        <p>Context Retrieval: {formatTime(message.timing.retrieval_time || 0)}</p>
                        <p>Answer Generation: {formatTime(message.timing.generation_time || 0)}</p>
                        <p className="font-medium">Total Time: {formatTime(message.timing.total_time || 0)}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {messages.length === 0 && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center space-y-6 max-w-lg">
                  <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-600 to-green-600 bg-clip-text text-transparent">
                    Welcome to NICE Guidelines
                  </h1>
                  <p className="text-xl text-gray-600">
                    Ask me anything about medical guidelines. I'm here to help!
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="border-t bg-white/60 backdrop-blur-lg p-6">
            <form onSubmit={handleSubmit} className="flex gap-4 max-w-4xl mx-auto">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about NICE guidelines..."
                disabled={isLoading}
                className="flex-1 bg-transparent border-2 text-black border-cyan-200 focus:border-cyan-500 focus:outline-none rounded-xl py-4 px-6 text-lg shadow-md hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed resize-none"
                rows={4}
              />
              <button 
                type="submit" 
                disabled={isLoading}
                className="bg-gradient-to-r from-cyan-500 to-green-500 hover:from-cyan-600 hover:to-green-600 text-white shadow-lg hover:shadow-xl transition-all w-16 h-16 rounded-xl flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <div className="h-6 w-6 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Send className="h-6 w-6" />
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}