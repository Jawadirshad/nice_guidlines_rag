import React, { useState } from 'react';
import { BookOpen, X } from 'lucide-react';

interface ContextModalProps {
  context: string;
}

const ContextModal = ({ context }: ContextModalProps) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Context Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="mt-4 flex items-center gap-2 text-cyan-600 hover:text-cyan-700 transition-colors"
      >
        <BookOpen className="h-5 w-5" />
        <span>View Retrieved Context</span>
      </button>

      {/* Modal Overlay */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div 
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setIsOpen(false)}
          />

          {/* Modal Content */}
          <div className="relative w-full max-w-2xl max-h-[80vh] bg-white rounded-2xl shadow-xl m-4 overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                Retrieved Context
              </h3>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto max-h-[calc(80vh-4rem)]">
              <pre className="whitespace-pre-wrap text-sm bg-gray-50 p-4 rounded-lg">
                {context}
              </pre>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ContextModal;