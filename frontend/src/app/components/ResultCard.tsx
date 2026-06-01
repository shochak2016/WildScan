import { ThreatBadge } from './ThreatBadge';
import { ScaleComparison } from './ScaleComparison';
import { Info, Shield, AlertTriangle, ThumbsUp, ThumbsDown } from 'lucide-react';

interface ResultCardProps {
  species: string;
  scientificName: string;
  confidence: number;
  threatLevel: 'safe' | 'caution' | 'dangerous';
  description: string;
  guidance: string[];
  habitat: string;
  onFeedback: (isCorrect: boolean) => void;
}

export function ResultCard({
  species,
  scientificName,
  confidence,
  threatLevel,
  description,
  guidance,
  habitat,
  onFeedback,
}: ResultCardProps) {
  const isDangerous = threatLevel === 'dangerous';
  const cardBg = isDangerous ? 'bg-red-900 border-red-600' : 'bg-white/80 backdrop-blur-sm border-emerald-200';
  const textColor = isDangerous ? 'text-red-50' : 'text-stone-900';
  const mutedTextColor = isDangerous ? 'text-red-200' : 'text-emerald-700';
  return (
    <div className={`rounded-2xl shadow-lg p-6 space-y-4 border transition-all duration-700 ${cardBg} relative`}>
      {isDangerous && (
        <div className="absolute inset-0 border-4 border-red-500 rounded-2xl animate-pulse pointer-events-none"></div>
      )}

      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <h2 className={`text-2xl font-bold ${textColor}`}>
            {species}
          </h2>
          <p className={`text-sm italic ${mutedTextColor}`}>{scientificName}</p>
        </div>
        <ThreatBadge level={threatLevel} />
      </div>

      <div className={`rounded-lg p-3 border ${isDangerous ? 'bg-red-950 border-red-700' : 'bg-stone-50 border-stone-200'}`}>
        <div className="flex items-center justify-between">
          <span className={`text-sm ${isDangerous ? 'text-red-200' : 'text-stone-700'}`}>
            Confidence
          </span>
          <span className={`font-semibold ${textColor}`}>{confidence}%</span>
        </div>
        <div className={`mt-2 h-2 rounded-full overflow-hidden ${isDangerous ? 'bg-red-800' : 'bg-stone-200'}`}>
          <div
            className={`h-full transition-all duration-500 ${isDangerous ? 'bg-red-400' : 'bg-stone-700'}`}
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>

      <ScaleComparison species={species} threatLevel={threatLevel} />

      <div className="space-y-3">
        <div className="flex items-start gap-3">
          <Info className={`w-5 h-5 flex-shrink-0 mt-0.5 ${isDangerous ? 'text-red-300' : 'text-blue-600'}`} />
          <div>
            <h3 className={`font-semibold mb-1 ${textColor}`}>Description</h3>
            <p className={`text-sm ${isDangerous ? 'text-red-100' : 'text-stone-700'}`}>{description}</p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <Shield className={`w-5 h-5 flex-shrink-0 mt-0.5 ${isDangerous ? 'text-red-300' : 'text-green-600'}`} />
          <div>
            <h3 className={`font-semibold mb-1 ${textColor}`}>Habitat</h3>
            <p className={`text-sm ${isDangerous ? 'text-red-100' : 'text-stone-700'}`}>{habitat}</p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <AlertTriangle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${isDangerous ? 'text-red-200 animate-pulse' : 'text-amber-600'}`} />
          <div className="flex-1">
            <h3 className={`font-semibold mb-2 ${textColor}`}>Safety Guidance</h3>
            <ul className="space-y-1.5">
              {guidance.map((item, index) => (
                <li key={index} className={`text-sm flex items-start gap-2 ${isDangerous ? 'text-red-100' : 'text-stone-700'}`}>
                  <span className={isDangerous ? 'text-red-300' : 'text-stone-500'}>•</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Feedback Section */}
      <div className={`border-t pt-4 ${isDangerous ? 'border-red-700' : 'border-stone-200'}`}>
        <p className={`text-sm mb-2 ${mutedTextColor}`}>Is this identification correct?</p>
        <div className="flex gap-2">
          <button
            onClick={() => onFeedback(true)}
            className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-lg transition-colors ${
              isDangerous
                ? 'bg-red-800 hover:bg-red-700 text-red-100'
                : 'bg-green-100 hover:bg-green-200 text-green-800'
            }`}
          >
            <ThumbsUp className="w-4 h-4" />
            <span className="text-sm font-medium">Correct</span>
          </button>
          <button
            onClick={() => onFeedback(false)}
            className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-lg transition-colors ${
              isDangerous
                ? 'bg-red-800 hover:bg-red-700 text-red-100'
                : 'bg-red-100 hover:bg-red-200 text-red-800'
            }`}
          >
            <ThumbsDown className="w-4 h-4" />
            <span className="text-sm font-medium">Incorrect</span>
          </button>
        </div>
      </div>
    </div>
  );
}
