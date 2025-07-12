import React, { useState, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { Scatter } from 'react-chartjs-2';
import Papa from 'papaparse';
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  Title,
  Filler,
  CategoryScale,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(LinearScale, PointElement, Tooltip, Legend, Title, Filler, CategoryScale, zoomPlugin);

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
  const [dataRows, setDataRows] = useState([]);
  const [selectedFields, setSelectedFields] = useState([]);
  const [showDialog, setShowDialog] = useState(false);
  const [mode, setMode] = useState('select');
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [selectedPointPosition, setSelectedPointPosition] = useState(null);
  const chartRefs = useRef([]);

  const { getRootProps, getInputProps } = useDropzone({
    accept: { 'text/csv': ['.csv'] },
    onDrop: acceptedFiles => {
      const file = acceptedFiles[0];
      const reader = new FileReader();
      reader.onload = () => {
        Papa.parse(reader.result, {
          header: true,
          skipEmptyLines: true,
          complete: (result) => {
            const fields = Object.keys(result.data[0]);
            setHeaders(fields);
            setDataRows(result.data);
            setShowDialog(true);
          }
        });
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

  const exportChartImage = (index) => {
    const chartInstance = chartRefs.current[index];
    if (chartInstance && chartInstance.canvas) {
      const originalCanvas = chartInstance.canvas;
      const width = originalCanvas.width;
      const height = originalCanvas.height;

      const exportCanvas = document.createElement("canvas");
      exportCanvas.width = width;
      exportCanvas.height = height;
      const ctx = exportCanvas.getContext("2d");

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      ctx.drawImage(originalCanvas, 0, 0);

      const url = exportCanvas.toDataURL("image/png");
      const link = document.createElement("a");
      link.download = `chart-${index + 1}.png`;
      link.href = url;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const pairs = generatePairs(selectedFields);

  return (
    <div className="flex flex-col items-center justify-between min-h-screen text-center bg-gray-100 relative">
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
          <div className="flex justify-start mb-4">
            <button
              onClick={() => setMode('select')}
              className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300"
            >
              🔙 Back to field selection
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {pairs.map(([xField, yField], index) => {
              const chartData = {
                datasets: [
                  {
                    label: `${xField} × ${yField}`,
                    data: dataRows.map(row => {
                      const point = {
                        x: parseFloat(row[xField]),
                        y: parseFloat(row[yField]),
                        full: row
                      };
                      return (!isNaN(point.x) && !isNaN(point.y)) ? point : null;
                    }).filter(p => p !== null),
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                  }
                ]
              };

              const chartOptions = {
                responsive: true,
                onClick: (event, elements, chart) => {
                  if (elements.length > 0) {
                    const datasetIndex = elements[0].datasetIndex;
                    const pointIndex = elements[0].index;
                    const pointData = chartData.datasets[datasetIndex].data[pointIndex];
                    setSelectedPoint(pointData.full);

                    const canvasRect = chart.canvas.getBoundingClientRect();
                    const pointElement = chart.getDatasetMeta(datasetIndex).data[pointIndex];
                    const x = canvasRect.left + pointElement.x;
                    const y = canvasRect.top + pointElement.y;
                    setSelectedPointPosition({ x, y });
                  }
                },
                plugins: {
                  legend: { position: 'top' },
                  tooltip: {
                    callbacks: {
                      label: context => {
                        const point = context.raw;
                        return `${xField}: ${point.x}, ${yField}: ${point.y}`;
                      }
                    }
                  },
                  zoom: {
                    pan: {
                      enabled: true,
                      mode: 'xy',
                    },
                    zoom: {
                      wheel: {
                        enabled: true,
                      },
                      pinch: {
                        enabled: true,
                      },
                      mode: 'xy',
                    },
                  },
                },
                scales: {
                  x: {
                    title: { display: true, text: xField },
                    beginAtZero: false
                  },
                  y: {
                    title: { display: true, text: yField },
                    beginAtZero: false
                  }
                }
              };

              return (
                <div
                  key={index}
                  className="relative bg-white rounded-lg shadow p-4 border border-gray-200 flex flex-col justify-between"
                >
                  <div className="flex justify-between items-center mb-2">
                    <h2 className="text-lg font-medium">
                      Graph {index + 1}: {xField} × {yField}
                    </h2>
                  </div>
                  <Scatter
                    data={chartData}
                    options={chartOptions}
                    ref={(el) => chartRefs.current[index] = el?.chartInstance || el?.chart || el}
                  />
                  <button
                    onClick={() => exportChartImage(index)}
                    className="mt-2 self-end px-2 py-1 text-xs bg-gray-300 text-gray-800 rounded hover:bg-gray-400 opacity-80"
                  >
                    📥 Download as PNG
                  </button>
                </div>
              );
            })}
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
                  setMode('plot');
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      {selectedPoint && selectedPointPosition && (
        <div
          className="absolute z-50 bg-white text-left border border-gray-300 shadow-lg rounded-md p-3 max-h-40 overflow-y-auto text-sm"
          style={{ top: selectedPointPosition.y - 60, left: selectedPointPosition.x + 10 }}
        >
          <div className="flex justify-end mb-1">
            <button
              onClick={() => setSelectedPoint(null)}
              className="text-xs text-blue-500 hover:underline"
            >
              Close
            </button>
          </div>
          {Object.entries(selectedPoint).map(([key, val]) => (
            <p key={key}><strong>{key}:</strong> {val}</p>
          ))}
        </div>
      )}
    </div>
  );
}

export default HomePage;
