import { useState } from 'react';
import { UploadSection } from './components/UploadSection';
import { ResultCard } from './components/ResultCard';
import { ScanHistory } from './components/ScanHistory';
import { Loader2, Camera, Clock } from 'lucide-react';

type Tab = 'scan' | 'history';

// Mock ML classification data
const mockClassifications = {
  image: [
    {
      species: 'Black Bear',
      scientificName: 'Ursus americanus',
      confidence: 94,
      threatLevel: 'caution' as const,
      description: 'Medium-sized bear native to North America. Generally shy but can be dangerous if surprised or protecting cubs.',
      habitat: 'Forests, mountains, and wooded areas across North America',
      guidance: [
        'Make yourself appear large and speak firmly',
        'Do not run - back away slowly while facing the bear',
        'If attacked, fight back aggressively using any available objects',
        'Never get between a mother and her cubs',
      ],
    },
    {
      species: 'White-tailed Deer',
      scientificName: 'Odocoileus virginianus',
      confidence: 97,
      threatLevel: 'safe' as const,
      description: 'Common herbivorous mammal found throughout the Americas. Generally harmless unless cornered.',
      habitat: 'Forests, grasslands, and suburban areas',
      guidance: [
        'Observe from a safe distance',
        'Do not attempt to feed or touch',
        'Give extra space during rutting season (fall)',
        'Watch for sudden movements if deer feels cornered',
      ],
    },
  ],
  audio: [
    {
      species: 'Coyote',
      scientificName: 'Canis latrans',
      confidence: 89,
      threatLevel: 'caution' as const,
      description: 'Highly adaptable canid found across North America. Usually avoid humans but may become bold in urban areas.',
      habitat: 'Diverse habitats including forests, prairies, and urban areas',
      guidance: [
        'Make loud noises and wave your arms to appear threatening',
        'Do not run or turn your back',
        'Keep pets on leash and close to you',
        'If approached, throw objects and shout aggressively',
      ],
    },
    {
      species: 'Great Horned Owl',
      scientificName: 'Bubo virginianus',
      confidence: 92,
      threatLevel: 'safe' as const,
      description: 'Large nocturnal bird of prey with distinctive ear tufts. Generally not dangerous to humans.',
      habitat: 'Widespread across Americas in forests, deserts, and urban areas',
      guidance: [
        'Enjoy from a distance',
        'Do not approach nests, especially during breeding season',
        'Use a flashlight if hiking at night to avoid startling',
        'Generally harmless unless defending nest or young',
      ],
    },
  ],
};

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('scan');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<typeof mockClassifications.image[0] | null>(null);
  const [uploadedFile, setUploadedFile] = useState<{ url: string; type: string } | null>(null);
  const [history, setHistory] = useState<Array<typeof mockClassifications.image[0] & { timestamp: Date }>>([]);

  const handleUpload = async (file: File, type: 'image' | 'audio') => {
    setIsProcessing(true);
    setResult(null);

    // Create preview URL
    const url = URL.createObjectURL(file);
    setUploadedFile({ url, type: file.type });

    // Simulate ML processing delay
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Mock classification result
    const classifications = type === 'image' ? mockClassifications.image : mockClassifications.audio;
    const randomResult = classifications[Math.floor(Math.random() * classifications.length)];

    setResult(randomResult);
    setHistory((prev) => [{ ...randomResult, timestamp: new Date() }, ...prev]);
    setIsProcessing(false);
  };

  const handleClearHistory = () => {
    if (confirm('Clear all scan history?')) {
      setHistory([]);
    }
  };

  const handleFeedback = (isCorrect: boolean) => {
    if (isCorrect) {
      alert('Thank you for confirming! This helps improve our model.');
    } else {
      alert('Thanks for the feedback. You can submit the correct species to help us improve.');
    }
  };

  const handleReset = () => {
    setResult(null);
    setUploadedFile(null);
    setIsProcessing(false);
    setActiveTab('scan');
  };

  // Dynamic theme based on threat level
  const getThemeClasses = () => {
    if (!result) return 'bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50';

    switch (result.threatLevel) {
      case 'safe':
        return 'bg-gradient-to-br from-emerald-50 via-green-50 to-blue-50';
      case 'caution':
        return 'bg-gradient-to-b from-amber-50 to-orange-100';
      case 'dangerous':
        return 'bg-gradient-to-b from-red-950 to-black';
      default:
        return 'bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50';
    }
  };

  const getHeaderTextColor = () => {
    if (result?.threatLevel === 'dangerous') return 'text-red-100';
    return 'text-stone-800';
  };

  return (
    <div className={`min-h-screen transition-all duration-700 ${getThemeClasses()} relative overflow-hidden`}>
      {/* Nature-inspired background pattern */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="leaves" x="0" y="0" width="200" height="200" patternUnits="userSpaceOnUse">
              <path d="M50,100 Q30,80 50,60 Q70,80 50,100 Z" fill="currentColor" className="text-emerald-800" />
              <path d="M150,50 Q130,30 150,10 Q170,30 150,50 Z" fill="currentColor" className="text-teal-800" />
              <path d="M100,180 Q80,160 100,140 Q120,160 100,180 Z" fill="currentColor" className="text-green-800" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#leaves)" />
        </svg>
      </div>

      <div className="relative max-w-2xl mx-auto px-4 py-8 space-y-6">
        {/* Header */}
        <div className="text-center space-y-3 py-4">
          <div className="flex items-center justify-center gap-3 mb-2">
            <h1 className={`text-5xl font-bold transition-colors duration-700 ${getHeaderTextColor()}`}>
              WildScan
            </h1>
          </div>
          <p className={`text-lg transition-colors duration-700 ${result?.threatLevel === 'dangerous' ? 'text-red-200' : 'text-emerald-700'}`}>
            Identify animals and assess safety instantly
          </p>
        </div>

        {/* Navigation Tabs */}
        {!result && (
          <div className="flex gap-2 bg-white/80 backdrop-blur-sm rounded-xl p-1 shadow-sm border border-emerald-200/50">
            <button
              onClick={() => setActiveTab('scan')}
              className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-lg transition-all ${
                activeTab === 'scan'
                  ? 'bg-gradient-to-r from-emerald-700 to-teal-700 text-white shadow-md'
                  : 'text-stone-700 hover:bg-emerald-50'
              }`}
            >
              <Camera className="w-4 h-4" />
              <span className="font-medium">Scan</span>
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-lg transition-all ${
                activeTab === 'history'
                  ? 'bg-gradient-to-r from-emerald-700 to-teal-700 text-white shadow-md'
                  : 'text-stone-700 hover:bg-emerald-50'
              }`}
            >
              <Clock className="w-4 h-4" />
              <span className="font-medium">History</span>
              {history.length > 0 && (
                <span className="bg-emerald-600 text-white text-xs px-2 py-0.5 rounded-full">
                  {history.length}
                </span>
              )}
            </button>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'scan' && !result && !isProcessing && (
          <UploadSection onUpload={handleUpload} isProcessing={isProcessing} />
        )}

        {activeTab === 'history' && !result && (
          <ScanHistory history={history} onClear={handleClearHistory} />
        )}

        {/* Processing State */}
        {isProcessing && (
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg p-12 text-center space-y-4 border border-emerald-200">
            {uploadedFile && (
              <div className="mb-4">
                {uploadedFile.type.startsWith('image/') ? (
                  <img
                    src={uploadedFile.url}
                    alt="Uploaded"
                    className="max-h-48 mx-auto rounded-lg shadow-md"
                  />
                ) : (
                  <audio src={uploadedFile.url} controls className="mx-auto" />
                )}
              </div>
            )}
            <Loader2 className="w-12 h-12 text-emerald-700 animate-spin mx-auto" />
            <p className="text-lg font-semibold text-stone-900">
              Analyzing with AI...
            </p>
            <p className="text-sm text-emerald-700">
              Running CNN classification model
            </p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-4">
            {uploadedFile && (
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg p-4 border border-emerald-200">
                {uploadedFile.type.startsWith('image/') ? (
                  <img
                    src={uploadedFile.url}
                    alt="Uploaded"
                    className="max-h-64 mx-auto rounded-lg shadow-md"
                  />
                ) : (
                  <audio src={uploadedFile.url} controls className="w-full" />
                )}
              </div>
            )}
            <ResultCard {...result} onFeedback={handleFeedback} />
            <button
              onClick={handleReset}
              className={`w-full font-semibold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-xl ${
                result.threatLevel === 'dangerous'
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-gradient-to-r from-emerald-700 to-teal-700 hover:from-emerald-800 hover:to-teal-800 text-white'
              }`}
            >
              Scan Another Animal
            </button>
          </div>
        )}

        {/* Info Footer */}
        {activeTab === 'scan' && (
          <div className={`text-center text-xs pt-4 transition-colors duration-700 ${
            result?.threatLevel === 'dangerous' ? 'text-red-300' : 'text-emerald-600'
          }`}>
            <p>Currently using mock ML models for demonstration</p>
          </div>
        )}
      </div>
    </div>
  );
}