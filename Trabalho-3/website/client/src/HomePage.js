import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';

function generatePairs(fields) {
  const pairs = [];
  for (let i = 0; i < fields.length; i++) {
    for (let j = i + 1; j < fields.length; j++) {
      pairs.push([fields[i], fields[j]]);
    }
  }
  return pairs;
}

function HomePage() {
  const [headers, setHeaders] = useState([]);
  const [selectedFields, setSelectedFields] = useState([]);
  const [showDialog, setShowDialog] = useState(false);
  const [mode, setMode] = useState('select');

  const { getRootProps, getInputProps } = useDropzone({
    accept: { 'text/csv': ['.csv'] },
    onDrop: acceptedFiles => {
      const file = acceptedFiles[0];
      const reader = new FileReader();
      reader.onload = () => {
        const lines = reader.result.split('\n');
        const headerLine = lines[0].trim();
        const fields = headerLine.split(',').map(h => h.trim());
        setHeaders(fields);
        setShowDialog(true);
      };
      reader.readAsText(file);
    }
  });

  const toggleField = (field) => {
    setSelectedFields(prev =>
      prev.includes(field)
        ? prev.filter(f => f !== field)
        : [...prev, field]
    );
  };

  const pairs = generatePairs(selectedFields);

  return (
    <div className="flex flex-col items-center justify-between min-h-screen text-center bg-gray-100">
      <header className="w-full py-8 border-b bg-white border-gray-300 shadow">
        <h1 className="text-4xl font-bold">PlotIt! 📊</h1>
      </header>

      {mode === 'select' && (
        <main className="flex flex-col items-center justify-center flex-grow">
          <p className="mb-4 text-2xl">Place your CSV and we will <strong>PlotIt!</strong></p>
          <div
            {...getRootProps()}
            className="p-4 border border-dashed border-gray-400 rounded-md cursor-pointer hover:bg-gray-200 bg-white"
          >
            <input {...getInputProps()} />
            <p>📁 Choose your csv file</p>
          </div>
        </main>
      )}

      {mode === 'plot' && (
        <main className="flex-1 w-full p-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {pairs.map(([x, y], index) => (
              <div
                key={index}
                className="bg-white rounded-lg shadow p-4 border border-gray-200 flex flex-col justify-between"
              >
                <h2 className="text-lg font-medium mb-2">
                  Graph {index + 1}: {x} × {y}
                </h2>
                <div className="flex-grow flex items-center justify-center border border-dashed border-gray-300 rounded-md text-gray-400">
                  (Graph placeholder)
                </div>
              </div>
            ))}
          </div>
        </main>
      )}

      <footer className="w-full py-6 border-t bg-white border-gray-300" />

      {showDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg w-[90%] max-w-md p-6 text-left">
            <h2 className="text-xl font-semibold mb-4">What information do you want to plot?</h2>
            <div className="max-h-60 overflow-y-auto space-y-2">
              {headers.map((field) => (
                <label key={field} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={selectedFields.includes(field)}
                    onChange={() => toggleField(field)}
                    className="accent-blue-600"
                  />
                  <span>{field}</span>
                </label>
              ))}
            </div>
            <div className="mt-6 flex justify-end space-x-3">
              <button
                onClick={() => setShowDialog(false)}
                className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowDialog(false);
                  setMode('plot'); // muda para modo gráfico
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default HomePage;
