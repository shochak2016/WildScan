import { Camera, Mic, Upload } from 'lucide-react';
import { useState } from 'react';

interface UploadSectionProps {
  onUpload: (file: File, type: 'image' | 'audio') => void;
  isProcessing: boolean;
}

export function UploadSection({ onUpload, isProcessing }: UploadSectionProps) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFile = (file: File) => {
    if (file.type.startsWith('image/')) {
      onUpload(file, 'image');
    } else if (file.type.startsWith('audio/')) {
      onUpload(file, 'audio');
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  return (
    <div className="space-y-4">
      <div
        className={`border-2 border-dashed rounded-2xl p-8 transition-colors ${
          dragActive
            ? 'border-emerald-400 bg-emerald-50'
            : 'border-emerald-300 bg-white/80 backdrop-blur-sm'
        } ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="text-center space-y-4">
          <div className="flex justify-center">
            <div className="bg-gradient-to-br from-emerald-100 to-teal-100 p-4 rounded-full">
              <Upload className="w-12 h-12 text-emerald-700" />
            </div>
          </div>
          <div>
            <p className="text-lg font-semibold text-stone-900">
              Drop your file here
            </p>
            <p className="text-sm text-emerald-700 mt-1">
              or tap below to select
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <label className="cursor-pointer">
          <input
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileInput}
            className="hidden"
            disabled={isProcessing}
          />
          <div className={`bg-white/80 backdrop-blur-sm border-2 border-emerald-200 rounded-xl p-6 text-center hover:border-emerald-400 hover:shadow-lg hover:bg-gradient-to-br hover:from-emerald-50 hover:to-teal-50 transition-all ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}>
            <Camera className="w-8 h-8 text-emerald-700 mx-auto mb-2" />
            <p className="font-semibold text-stone-900">Photo</p>
            <p className="text-xs text-emerald-700 mt-1">Take or upload</p>
          </div>
        </label>

        <label className="cursor-pointer">
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileInput}
            className="hidden"
            disabled={isProcessing}
          />
          <div className={`bg-white/80 backdrop-blur-sm border-2 border-teal-200 rounded-xl p-6 text-center hover:border-teal-400 hover:shadow-lg hover:bg-gradient-to-br hover:from-teal-50 hover:to-cyan-50 transition-all ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}>
            <Mic className="w-8 h-8 text-teal-700 mx-auto mb-2" />
            <p className="font-semibold text-stone-900">Sound</p>
            <p className="text-xs text-teal-700 mt-1">Record or upload</p>
          </div>
        </label>
      </div>
    </div>
  );
}
