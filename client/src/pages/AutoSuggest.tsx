
interface AutoSuggestProps {
  suggestions: string[];
  onSelect: (suggestion: string) => void;
  onClose: () => void;
}

export const AUTO_SUGGESTIONS: Record<string, string[]> = {
  "Cancer Treatment Guidelines": [
    "What are the current NICE guidelines for treating breast cancer?",
    "Show me the latest NICE recommendations for lung cancer treatment",
    "What's the recommended treatment pathway for prostate cancer?",
    "Guidelines for managing metastatic colorectal cancer",
    "Current NICE guidance for treating melanoma"
  ],
  "Cancer Drug Information": [
    "What are the main side effects and contraindications of pembrolizumab?",
    "Show dosing guidelines for cisplatin in cancer patients",
    "What are the interactions between tamoxifen and common medications?",
    "Contraindications and precautions for immunotherapy in cancer",
    "Guidelines for prescribing chemotherapy in elderly patients"
  ],
  "Cancer Clinical Pathways": [
    "What's the NICE diagnostic pathway for suspected ovarian cancer?",
    "Show the management pathway for acute leukemia",
    "Clinical pathway for assessing pancreatic cancer",
    "Cancer referral pathways for primary care",
    "Pathway for managing cancer-related pain"
  ],
  "Cancer Best Practices": [
    "What are the current best practices for cancer pain management?",
    "Show guidelines for preventing chemotherapy-induced nausea",
    "Best practices for managing cancer-related fatigue",
    "Evidence-based strategies for cancer prevention",
    "Current recommendations for cancer survivorship care"
  ]
};

const AutoSuggest: React.FC<AutoSuggestProps> = ({ suggestions, onSelect, onClose }) => (
  <div className="absolute bottom-full left-0 right-0 mb-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 max-h-60 overflow-y-auto">
    <div className="p-2">
      <div className="flex justify-between items-center text-sm font-medium text-gray-500 dark:text-gray-400 mb-2 px-2">
        <span>Suggested queries:</span>
        <button 
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
        >
          ✕
        </button>
      </div>
      {suggestions.map((suggestion, index) => (
        <button
          key={index}
          onClick={() => onSelect(suggestion)}
          className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
        >
          {suggestion}
        </button>
      ))}
    </div>
  </div>
);


export default AutoSuggest;