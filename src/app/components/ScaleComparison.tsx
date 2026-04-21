interface ScaleComparisonProps {
  species: string;
  threatLevel: 'safe' | 'caution' | 'dangerous';
}

export function ScaleComparison({ species, threatLevel }: ScaleComparisonProps) {
  // Mock size data - in production, this would come from your database
  const sizeData: Record<string, { height: number; weight: string }> = {
    'Black Bear': { height: 180, weight: '200-600 lbs' },
    'White-tailed Deer': { height: 100, weight: '100-300 lbs' },
    'Coyote': { height: 60, weight: '20-50 lbs' },
    'Great Horned Owl': { height: 50, weight: '2-5 lbs' },
  };

  const animalSize = sizeData[species] || { height: 100, weight: 'Unknown' };
  const humanHeight = 170; // cm
  const scaleRatio = (animalSize.height / humanHeight) * 100;

  const isDangerous = threatLevel === 'dangerous';

  return (
    <div className={`rounded-lg p-4 border ${isDangerous ? 'bg-red-950 border-red-700' : 'bg-stone-50 border-stone-200'}`}>
      <h3 className={`font-semibold mb-3 ${isDangerous ? 'text-red-50' : 'text-stone-900'}`}>
        Scale Comparison
      </h3>
      <div className="flex items-end justify-center gap-8 h-32">
        {/* Human silhouette */}
        <div className="flex flex-col items-center">
          <div className="relative h-24 w-12 bg-stone-400 rounded-t-full flex items-center justify-center">
            <div className="absolute top-0 w-8 h-8 bg-stone-400 rounded-full -translate-y-4"></div>
          </div>
          <span className={`text-xs mt-2 ${isDangerous ? 'text-red-200' : 'text-stone-600'}`}>Human (5'7")</span>
        </div>

        {/* Animal silhouette */}
        <div className="flex flex-col items-center">
          <div
            className={`relative rounded-t-full ${isDangerous ? 'bg-red-400' : 'bg-blue-500'}`}
            style={{
              height: `${Math.max(scaleRatio, 20)}%`,
              width: '48px',
              maxHeight: '96px',
            }}
          >
            <div
              className={`absolute top-0 rounded-full ${isDangerous ? 'bg-red-400' : 'bg-blue-500'}`}
              style={{
                width: '32px',
                height: '32px',
                transform: 'translateY(-50%)',
              }}
            ></div>
          </div>
          <span className={`text-xs mt-2 ${isDangerous ? 'text-red-200' : 'text-stone-600'}`}>
            {species}
          </span>
        </div>
      </div>
      <div className={`text-center mt-4 pt-3 border-t ${isDangerous ? 'border-red-800 text-red-200' : 'border-stone-200 text-stone-600'}`}>
        <p className="text-sm">
          Avg. Height: {animalSize.height}cm | Weight: {animalSize.weight}
        </p>
      </div>
    </div>
  );
}
