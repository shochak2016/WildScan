import { ThreatBadge } from './ThreatBadge';
import { Calendar, MapPin, Trash2 } from 'lucide-react';

interface HistoryEntry {
  species: string;
  scientificName: string;
  threatLevel: 'safe' | 'caution' | 'dangerous';
  confidence: number;
  timestamp: Date;
  location?: string;
}

interface ScanHistoryProps {
  history: HistoryEntry[];
  onClear: () => void;
}

export function ScanHistory({ history, onClear }: ScanHistoryProps) {
  if (history.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-stone-800 mb-2">
            Explorer's Log
          </h2>
          <p className="text-stone-600">Your wildlife sighting history</p>
        </div>

        <div className="bg-stone-50 border border-stone-200 rounded-lg p-12 text-center">
          <div className="text-stone-400 mb-4">
            <Calendar className="w-16 h-16 mx-auto" />
          </div>
          <h3 className="font-semibold text-stone-900 mb-2">
            No scans yet
          </h3>
          <p className="text-sm text-stone-600">
            Start scanning animals to build your explorer's log
          </p>
        </div>
      </div>
    );
  }

  // Group by date
  const groupedHistory: Record<string, HistoryEntry[]> = {};
  history.forEach((entry) => {
    const dateKey = entry.timestamp.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
    if (!groupedHistory[dateKey]) {
      groupedHistory[dateKey] = [];
    }
    groupedHistory[dateKey].push(entry);
  });

  // Count unique species
  const uniqueSpecies = new Set(history.map((e) => e.species)).size;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-stone-800 mb-2">
            Explorer's Log
          </h2>
          <p className="text-stone-600">Your wildlife sighting history</p>
        </div>
        <button
          onClick={onClear}
          className="flex items-center gap-2 text-red-600 hover:text-red-700 transition-colors"
        >
          <Trash2 className="w-4 h-4" />
          <span className="text-sm font-medium">Clear All</span>
        </button>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <p className="text-sm text-blue-700">Total Scans</p>
          <p className="text-2xl font-bold text-blue-900 mt-1">
            {history.length}
          </p>
        </div>
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <p className="text-sm text-green-700">Unique Species</p>
          <p className="text-2xl font-bold text-green-900 mt-1">
            {uniqueSpecies}
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {Object.entries(groupedHistory).map(([date, entries]) => (
          <div key={date}>
            <div className="flex items-center gap-2 mb-3">
              <Calendar className="w-4 h-4 text-stone-500" />
              <h3 className="font-semibold text-stone-700">{date}</h3>
            </div>
            <div className="space-y-2">
              {entries.map((entry, index) => (
                <div
                  key={index}
                  className="bg-white border border-stone-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <h4 className="font-semibold text-stone-900">
                        {entry.species}
                      </h4>
                      <p className="text-sm text-stone-500 italic">
                        {entry.scientificName}
                      </p>
                      <div className="flex items-center gap-3 mt-2 text-xs text-stone-500">
                        <span>
                          {entry.timestamp.toLocaleTimeString('en-US', {
                            hour: 'numeric',
                            minute: '2-digit',
                          })}
                        </span>
                        {entry.location && (
                          <>
                            <span>•</span>
                            <span className="flex items-center gap-1">
                              <MapPin className="w-3 h-3" />
                              {entry.location}
                            </span>
                          </>
                        )}
                        <span>•</span>
                        <span>{entry.confidence}% confidence</span>
                      </div>
                    </div>
                    <ThreatBadge level={entry.threatLevel} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
