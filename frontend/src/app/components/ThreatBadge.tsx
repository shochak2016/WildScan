interface ThreatBadgeProps {
  level: 'safe' | 'caution' | 'dangerous';
}

import { Shield, AlertTriangle, AlertOctagon } from 'lucide-react';

export function ThreatBadge({ level }: ThreatBadgeProps) {
  const styles = {
    safe: 'bg-green-100 text-green-800 border-green-400',
    caution: 'bg-amber-100 text-amber-800 border-amber-400',
    dangerous: 'bg-red-600 text-white border-red-800 animate-pulse',
  };

  const icons = {
    safe: Shield,
    caution: AlertTriangle,
    dangerous: AlertOctagon,
  };

  const Icon = icons[level];

  return (
    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full border-2 ${styles[level]}`}>
      <Icon className="w-5 h-5" />
      <span className="font-semibold capitalize">{level}</span>
    </div>
  );
}
