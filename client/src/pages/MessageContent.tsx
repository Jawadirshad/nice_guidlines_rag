
import React, { useState, useEffect, useRef } from "react";
import { MessageCircle, Send, Clock, BookOpen, LinkIcon, ArrowUp, Sun, Moon, Copy, Check } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
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
  
  
  const MessageContent: React.FC<{ message: Message; theme: string }> = ({ message, theme }) => {
    const [copied, setCopied] = useState(false);
    const [displayedContent, setDisplayedContent] = useState('');
    const [isComplete, setIsComplete] = useState(false);
    const isAssistantMessage = message.role === 'assistant';
    const contentRef = useRef<HTMLDivElement>(null);
  
    useEffect(() => {
      if (!isAssistantMessage) {
        setDisplayedContent(message.content);
        setIsComplete(true);
        return;
      }
  
      let currentIndex = 0;
      const content = message.content;
      const typingSpeed = Math.max(10, Math.min(30, content.length / 100));
      
      const interval = setInterval(() => {
        if (currentIndex < content.length) {
          setDisplayedContent(prev => prev + (content[currentIndex] || ''));      
          
          currentIndex++;
        } else {
          clearInterval(interval);
          setIsComplete(true);
        }
      }, typingSpeed);
  
      return () => clearInterval(interval);
    }, [message.content, isAssistantMessage]);
  
    const handleCopy = async () => {
      if (!message.content) return;
  
  
  
      try {
  
  
  
        await navigator.clipboard.writeText(message.content).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        });
      } catch (err) {
        console.error('Clipboard API failed:', err);
        // Fallback to traditional method
        try {
          const textArea = document.createElement('textarea');
          textArea.value = message.content;
          textArea.style.position = 'fixed';
          textArea.style.opacity = '0';
          document.body.appendChild(textArea);
          textArea.select();
          document.execCommand('copy');
          document.body.removeChild(textArea);
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        } catch (fallbackErr) {
          console.error('Fallback copy failed:', fallbackErr);
        }
      }
    };
  
    return (
      <div className="group relative">
        <div className={`prose ${theme === 'dark' ? 'prose-invert' : ''} max-w-none`}>
          {displayedContent && (
            <ReactMarkdown
         
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({node, ...props}) => <h1 className={`text-2xl font-bold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`} {...props} />,
                h2: ({node, ...props}) => <h2 className={`text-xl font-bold mb-3 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`} {...props} />,
                h3: ({node, ...props}) => <h3 className={`text-lg font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`} {...props} />,
                p: ({node, ...props}) => <p className={`mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-700'}`} {...props} />,
                ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-4" {...props} />,
                ol: ({node, ...props}) => <ol className="list-decimal pl-5 mb-4" {...props} />,
                li: ({node, ...props}) => <li className="mb-1" {...props} />,
                code: ({node, inline, className, children, ...props}) => {
                  const match = /language-(\w+)/.exec(className || '');
                  const language = match ? match[1] : '';
                  
                  if (inline) {
                    return (
                      <code className={`px-1 py-0.5 rounded ${theme === 'dark' ? 'bg-gray-700 text-gray-200' : 'bg-gray-100 text-gray-800'}`} {...props}>
                        {children}
                      </code>
                    );
                  }
                  
                  return (
                    <pre className={`p-4 rounded-lg overflow-x-auto ${theme === 'dark' ? 'bg-gray-800' : 'bg-gray-50'}`}>
                      <code className={`language-${language}`} {...props}>
                        {children}
                      </code>
                    </pre>
                  );
                }
              }}
            >
              {displayedContent}
            </ReactMarkdown>
          )}
          {!isComplete && <CircularLoader />}
        </div>
        
        {message.role === 'assistant' && isComplete && (
          <button
            onClick={handleCopy}
            className="absolute right-2 p-2 rounded-lg bg-gray-100 dark:bg-gray-700 
                     opacity-0 group-hover:opacity-100 hover:bg-gray-200 dark:hover:bg-gray-600 
                     transition-all duration-200 flex items-center gap-2"
            title={copied ? "Copied!" : "Copy to clipboard"}
          >
            {copied ? (
              <>
                <Check className="h-4 w-4 text-green-500" />
                <span className="text-xs text-green-500 font-medium">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                <span className="text-xs text-gray-500 dark:text-gray-400">Copy</span>
              </>
            )}
          </button>
        )}
      </div>
    );
  };

  
  export default MessageContent;

