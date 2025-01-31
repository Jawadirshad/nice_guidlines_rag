import React, { useState, useEffect, useRef } from "react";
import { MessageCircle, Send, Clock, BookOpen, LinkIcon, ArrowUp, Sun, Moon, Copy, Check } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import AutoSuggest from "./AutoSuggest";
import MessageContent from "./MessageContent";
import { AUTO_SUGGESTIONS } from "./AutoSuggest";
const TypingAnimation = () => (
  <div className="flex space-x-2 p-2">
    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
  </div>
);

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

interface MessageContentProps {
  message: Message;
}



const CircularLoader = () => (
  <div className="flex justify-center items-center p-4">
    <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
  </div>
);


const SuggestedQueries: React.FC<{ onSelectQuery: (query: string) => void }> = ({ onSelectQuery }) => {
  const queries = [
    {
      title: "Cancer Treatment Guidelines",
      icon: "💊",
      description: "Find the latest NICE treatment recommendations for cancer"
    },
    {
      title: "Cancer Drug Information",
      icon: "💉",
      description: "Get detailed information about cancer medications"
    },
    {
      title: "Cancer Clinical Pathways",
      icon: "🔬",
      description: "View standard clinical pathways for cancer"
    },
    {
      title: "Cancer Best Practices",
      icon: "📋",
      description: "Learn about current best practices in cancer care"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl mx-auto p-4">
      {queries.map((query, index) => (
        <button
          key={index}
          onClick={() => onSelectQuery(query.title)}
          className="flex items-start gap-3 p-4 rounded-xl border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors text-left group"
        >
          <span className="text-2xl group-hover:scale-110 transition-transform">{query.icon}</span>
          <div>
            <h3 className="font-medium text-gray-900 dark:text-gray-100">{query.title}</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">{query.description}</p>
          </div>
        </button>
      ))}
    </div>
  );
};

const ContextDialog: React.FC<{ context: string }> = ({ context }) => (
  <Dialog>
    <DialogTrigger asChild>
      <Button variant="outline" className="mt-4 w-full md:w-1/2 group hover:bg-gray-100 dark:hover:bg-gray-800">
        <BookOpen className="h-4 w-4 mr-2 text-gray-600 dark:text-gray-300" />
        <span>View Medical Context</span>
      </Button>
    </DialogTrigger>
    <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>Clinical Context</DialogTitle>
      </DialogHeader>
      <div className="mt-4 text-sm whitespace-pre-wrap max-w-none">
        {context.split('\n').map((line, index) => (
          <p key={index} className="mb-2 last:mb-0">{line.trim()}</p>
        ))}
      </div>
    </DialogContent>
  </Dialog>
);

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [currentSuggestions, setCurrentSuggestions] = useState<string[]>([]);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window !== 'undefined') {
    
      const savedTheme = localStorage.getItem('theme') as 'light' | 'dark';
      
      console.log(savedTheme,"furqan");
      if (savedTheme){
        
        


        return savedTheme;



      }
           
      console.log(savedTheme,"furqan1");
      
      
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'light';
  });

  useEffect(() => {
    const handleScroll = () => setShowScrollTop(window.scrollY > 400);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);



    
  }, []);

  useEffect(() => {
    console.log("herer");
    
    document.documentElement.classList.toggle('dark', theme === 'dark');
    console.log(theme);

    localStorage.setItem('theme', theme);
    
  }, [theme]);

  const toggleTheme = () => {
  
    setTheme(current => current === 'light' ? 'dark' : 'light');
  };

  const API_BASE_URL = 'http://localhost:8000';

  const formatTime = (seconds: number) => `${seconds.toFixed(2)}s`;
  const scrollToTop = () => window.scrollTo({ top: 0, behavior: 'smooth' });

  const formatTimestamp = (timestamp: Date): string => {
    const now = new Date();
    const seconds = Math.floor((now.getTime() - timestamp.getTime()) / 1000);

    if (seconds < 10) return "Just now";
    if (seconds < 60) return `${seconds} seconds ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} hours ago`;
    return `${Math.floor(seconds / 86400)} days ago`;
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleQuerySelect = (query: string) => {
    const suggestions = AUTO_SUGGESTIONS[query] || [];
    setCurrentSuggestions(suggestions);
    setShowSuggestions(true);
  };

  const handleSuggestionSelect = async (suggestion: string) => {
    const userMessage: Message = { 
      role: "user", 
      content: suggestion,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage].slice(-16));
    setShowSuggestions(false);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: suggestion,
          conversation_history: messages.map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        })
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      
      const data = await response.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        context: data.context,
        references: data.results?.map(res => ({
          link: res.pdf_link?.includes('nice.org.uk') || res.pdf_link?.startsWith('www.nice.org.uk')
            ? `https://${res.pdf_link.replace(/^(?:https?:\/\/)?(?:www\.)?/, '')}`
            : res.pdf_link || ""
        })),
        timing: data.timing,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage].slice(-16));
    } catch (error) {
      console.error("API call failed:", error);
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "I apologize, but I'm having trouble processing your request. Please try again.",
        timestamp: new Date()
      }].slice(-16));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { 
      role: "user", 
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage].slice(-16));
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: userMessage.content,
          conversation_history: messages.map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        })
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      
      const data = await response.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        context: data.context,
        references: data.results?.map(res => ({
          link: res.pdf_link?.includes('nice.org.uk') || res.pdf_link?.startsWith('www.nice.org.uk')
            ? `https://${res.pdf_link.replace(/^(?:https?:\/\/)?(?:www\.)?/, '')}`
            : res.pdf_link || ""
        })),
        timing: data.timing,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage].slice(-16));
    } catch (error) {
      console.error("API call failed:", error);
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "I apologize, but I'm having trouble processing your request. Please try again.",
        timestamp: new Date()
      }].slice(-16));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-5xl mx-auto flex h-16 items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <MessageCircle className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            <div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                Medical Assistant
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                NICE Guidelines Expert
              </p>
            </div>
          </div>
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >

            
            {theme === 'light' ? (
              <Moon className="h-5 w-5 text-gray-600" />
            ) : theme ==='dark' ?  (
              <Sun className="h-5 w-5 text-gray-300" />
            ) : <></>}
          </button>
        </div>
      </header>

      {/* Chat Container */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto">
          {messages.length === 0 ? (
            <>
              <div className="flex items-center justify-center h-[40vh] px-4">
                <div className="text-center space-y-4">
                  <div className="flex justify-center items-center mb-8">
                    <MessageCircle className="h-16 w-16 text-blue-600 dark:text-blue-400" />
                  </div>
                  <h2 className="text-4xl font-bold text-gray-900 dark:text-gray-100">
                    Welcome to Your Medical Assistant
                  </h2>
                  <p className="text-xl text-gray-600 dark:text-gray-300">
                    Access evidence-based medical guidelines and recommendations
                  </p>
                </div>
              </div>
              <SuggestedQueries onSelectQuery={handleQuerySelect} />
            </>
          ) : (
            <div className="py-4 space-y-6">
              {messages.map((message, index) => (
                <div key={index} className="px-4">
                  <div className="flex items-start gap-4 max-w-3xl mx-auto">
                    <div className={`flex-1 ${message.role === "user" ? "ml-auto max-w-2xl" : "mr-auto max-w-2xl"}`}>
                      <div className={`rounded-lg p-4 ${
                        message.role === "user"
                          ? "bg-gray-300 dark:bg-gray-800 text-white"
                          : "bg-white dark:bg-gray-800 shadow-sm"
                      }`}>
                        <div className="flex justify-between items-center mb-2">
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            {message.role === "user" ? "You" : "Medical Assistant"}
                          </div>
                          <div className="text-xs text-gray-600 dark:text-gray-400">
                            {formatTimestamp(message.timestamp)}
                          </div>
                        </div>
                        
                        <MessageContent message={message} theme={theme} />

                        {message.context && <ContextDialog context={message.context} />}

                        {message.references?.length > 0 && (
                          <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                            <h3 className="text-sm font-medium flex items-center gap-2 mb-2">
                              <LinkIcon className="h-4 w-4" />
                              Clinical References
                            </h3>
                            <ul className="space-y-1 text-sm">
                              {message.references.map((ref, idx) => (
                                <li key={idx}>
                                  <a
                                    href={ref.link}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-500 dark:text-blue-400 hover:underline break-all"
                                  >
                                    {ref.link.replace(/^(?:https?:\/\/)?(?:www\.)?/, '')}
                                  </a>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {message.timing && (
                          <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                            <div className="text-xs space-y-1">
                              <div className="flex justify-between">
                                <span>Search Time:</span>
                                <span className="font-mono">{formatTime(message.timing.retrieval_time || 0)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Analysis Time:</span>
                                <span className="font-mono">{formatTime(message.timing.generation_time || 0)}</span>
                              </div>
                              <div className="flex justify-between font-medium">
                                <span>Total Time:</span>
                                <span className="font-mono">{formatTime(message.timing.total_time || 0)}</span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t bg-white dark:bg-gray-800">
        <div className="max-w-3xl mx-auto p-4">
          <form onSubmit={handleSubmit} className="relative">
            {showSuggestions && currentSuggestions.length > 0 && (
              <AutoSuggest
                suggestions={currentSuggestions}
                onSelect={handleSuggestionSelect}
                onClose={() => setShowSuggestions(false)}
              />
            )}
            <textarea
              ref={textAreaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about cancer guidelines, treatments, or best practices..."
              disabled={isLoading}
              className="w-full resize-none rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-white text-sm min-h-[56px] max-h-[200px] overflow-y-auto"
              rows={1}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="absolute right-3 bottom-3 p-1 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? (
                <div className="h-5 w-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </button>
          </form>
          <div className="mt-2 text-xs text-center text-gray-500 dark:text-gray-400 space-y-1">
            <div className="flex items-center justify-center gap-4">
              <span>Press Enter to send</span>
              <span className="text-gray-400">•</span>
              <span>Shift + Enter for new line</span>
            </div>
         
          </div>
        </div>
      </div>

      {showScrollTop && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-20 right-8 bg-gray-900 dark:bg-gray-700 text-white p-3 rounded-full shadow-lg hover:bg-gray-800 dark:hover:bg-gray-600 transition-colors"
          aria-label="Scroll to top"
        >
          <ArrowUp className="h-5 w-5" />
        </button>
      )}
    </div>
  );
};

export default Chat;