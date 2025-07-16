import React, { useState, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Scatter } from 'react-chartjs-2';
import Papa from 'papaparse';
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  Filler,
  CategoryScale,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';

// Register Chart.js components and plugins
ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend, Title, Filler, CategoryScale, zoomPlugin);

// --- Helper Functions ---

// Creates a color map for unique values in a field
function getColorMap(values) {
  const colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];
  const uniqueValues = [...new Set(values)].filter(v => v !== undefined && v !== '');
  const map = {};
  uniqueValues.forEach((v, i) => {
    map[v] = colors[i % colors.length];
  });
  return map;
}

// Generates a composite key from a row based on specified fields
const getCompositeKey = (row, fields) => fields.map(field => row[field]).join(' - ');

// Generates pairs of fields for plotting
function generatePairs(fields) {
  const pairs = [];
  for (let i = 0; i < fields.length; i++) {
    for (let j = i + 1; j < fields.length; j++) {
      pairs.push([fields[i], fields[j]]);
    }
  }
  return pairs;
}

// Finds the Pareto frontier from a set of points for different optimization goals
function findParetoFrontier(points, direction = 'top-left') {
    if (!points || points.length === 0) return [];

    // Sort points based on the chosen direction
    const sortedPoints = [...points].sort((a, b) => {
        switch (direction) {
            case 'top-right': // Maximize X, Maximize Y
            case 'bottom-right': // Maximize X, Minimize Y
                if (a.x < b.x) return 1;
                if (a.x > b.x) return -1;
                return direction === 'bottom-right' ? (a.y < b.y ? -1 : 1) : (a.y > b.y ? -1 : 1);
            case 'top-left': // Minimize X, Maximize Y
            case 'bottom-left': // Minimize X, Minimize Y
            default:
                if (a.x < b.x) return -1;
                if (a.x > b.x) return 1;
                return direction === 'bottom-left' ? (a.y < b.y ? -1 : 1) : (a.y > b.y ? -1 : 1);
        }
    });

    const paretoFrontier = [];
    if (direction === 'top-left' || direction === 'top-right') {
        let maxY = -Infinity;
        for (const point of sortedPoints) {
            if (point.y > maxY) {
                paretoFrontier.push(point);
                maxY = point.y;
            }
        }
    } else { // bottom-left or bottom-right
        let minY = Infinity;
        for (const point of sortedPoints) {
            if (point.y < minY) {
                paretoFrontier.push(point);
                minY = point.y;
            }
        }
    }
    
    if (direction.includes('right')) {
        return paretoFrontier.sort((a,b) => a.x - b.x);
    }
    
    return paretoFrontier;
}

// --- ScrollableLegend Component ---
function ScrollableLegend({ datasets, fullscreen }) {
    const legendItems = datasets.filter(ds => ds.type === 'scatter' && ds.label && ds.label !== 'N/A');

    if (legendItems.length === 0) {
        return null;
    }

    const maxHeight = fullscreen ? 500 : 250;

    return (
        <div className="flex-shrink-0 w-48 ml-4 pr-2">
            <h4 className="text-sm font-semibold mb-2 border-b pb-1">Legend</h4>
            <div className={`max-h-[${maxHeight}px] overflow-y-auto custom-scrollbar`}>
                {legendItems.map(ds => (
                    <div key={ds.label} className="flex items-center mb-1 text-xs">
                        <span
                            className="w-3 h-3 inline-block mr-2 rounded-sm"
                            style={{ backgroundColor: ds.backgroundColor }}
                        ></span>
                        <span className="truncate">{ds.label}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}


// --- ChartWrapper Component ---
function ChartWrapper({
  dataRows,
  xField,
  yField,
  colorFields,
  allHeaders,
  colorMap,
  onPointClick,
  onFullscreen,
  onExport,
  chartRef,
  selectedPoint,
  selectedPointPosition,
  onClosePopup
}) {
  const [filterField, setFilterField] = useState('');
  const [filterValue, setFilterValue] = useState('');
  const [uniqueFilterValues, setUniqueFilterValues] = useState([]);
  const [showPareto, setShowPareto] = useState(false);
  const [paretoDirection, setParetoDirection] = useState('top-left');
  const [aggregationMethod, setAggregationMethod] = useState('none'); // 'none', 'average', 'sum'

  const availableFilterFields = allHeaders.filter(
    h => h !== xField && h !== yField
  );

  useEffect(() => {
    if (filterField) {
      const values = [...new Set(dataRows.map(row => row[filterField]))].sort();
      setUniqueFilterValues(values);
      setFilterValue('');
    } else {
      setUniqueFilterValues([]);
      setFilterValue('');
    }
  }, [filterField, dataRows]);

  const filteredData = (filterField && filterValue)
    ? dataRows.filter(row => String(row[filterField]) === String(filterValue))
    : dataRows;

  // --- Data Aggregation and Dataset Preparation ---
  const categories = [...new Set(filteredData.map(row => getCompositeKey(row, colorFields)))];
  const scatterDatasets = categories.map((category) => {
    const categoryPointsRaw = filteredData
      .filter(row => getCompositeKey(row, colorFields) === category)
      .map(row => {
        const x = parseFloat(row[xField]);
        const y = parseFloat(row[yField]);
        return (!isNaN(x) && !isNaN(y)) ? { x, y, full: row } : null;
      })
      .filter(p => p !== null);

    let finalPoints = [];

    if (aggregationMethod === 'none' || categoryPointsRaw.length === 0) {
        finalPoints = categoryPointsRaw;
    } else {
        const numPoints = categoryPointsRaw.length;
        const sumX = categoryPointsRaw.reduce((acc, p) => acc + p.x, 0);
        const sumY = categoryPointsRaw.reduce((acc, p) => acc + p.y, 0);

        if (aggregationMethod === 'average') {
            const avgX = sumX / numPoints;
            const avgY = sumY / numPoints;
            finalPoints = [{
                x: avgX,
                y: avgY,
                full: { _aggregated: true, _type: 'Average', _count: numPoints, [xField]: avgX, [yField]: avgY }
            }];
        } else if (aggregationMethod === 'sum') {
            finalPoints = [{
                x: sumX,
                y: sumY,
                full: { _aggregated: true, _type: 'Sum', _count: numPoints, [xField]: sumX, [yField]: sumY }
            }];
        }
    }

    return {
      label: category || 'N/A',
      data: finalPoints,
      backgroundColor: colorMap[category],
      type: 'scatter',
    };
  });
  
  const allDatasets = [...scatterDatasets];

  if (showPareto) {
      const allPointsForPareto = scatterDatasets.flatMap(dataset => dataset.data);

      const paretoFrontier = findParetoFrontier(allPointsForPareto, paretoDirection);
      const paretoDataset = {
          label: `Pareto Frontier`,
          data: paretoFrontier,
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 2,
          pointRadius: 0,
          showLine: true,
          fill: false,
          type: 'line',
      };
      allDatasets.push(paretoDataset);
  }

  const chartData = { datasets: allDatasets };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    onClick: (event, elements, chart) => {
      if (elements.length > 0) {
        const { datasetIndex, index } = elements[0];
        const dataset = chart.data.datasets[datasetIndex];
        if (dataset.type === 'scatter') {
            const pointData = dataset.data[index];
            if (pointData.full && !pointData.full._aggregated) {
                onPointClick(pointData.full, chart, elements[0]);
            }
        }
      }
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: context => {
            if (context.dataset.type === 'line') return null;
            const point = context.raw;
            const fullData = point.full;

            if (fullData && fullData._aggregated) {
                return [
                    `${fullData._type} of ${fullData._count} points`,
                    `${xField}: ${fullData[xField].toFixed(2)}`,
                    `${yField}: ${fullData[yField].toFixed(2)}`
                ];
            }
            
            const colorValue = getCompositeKey(fullData, colorFields);
            return `${xField}: ${point.x}, ${yField}: ${point.y}` + (colorFields.length ? `, ${colorValue}` : '');
          }
        }
      },
      zoom: {
        pan: { enabled: true, mode: 'xy' },
        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' },
      },
    },
    scales: {
      x: { title: { display: true, text: xField }, beginAtZero: false },
      y: { title: { display: true, text: yField }, beginAtZero: false }
    }
  };

  return (
    <div className="relative bg-white rounded-lg shadow p-4 border border-gray-200 flex flex-col justify-between h-[500px] chart-wrapper-div">
      <button onClick={onFullscreen} className="text-sm text-black hover:underline absolute top-2 right-2 z-10">
        🔍 Fullscreen
      </button>
      <h2 className="text-lg font-medium text-left mb-2">{xField} vs {yField}</h2>

      {/* Controls Container */}
      <div className="flex flex-col gap-2 mb-2">
        <div className="flex gap-2 items-center">
          <select value={filterField} onChange={(e) => setFilterField(e.target.value)} className="p-1 border rounded text-sm w-1/2">
            <option value="">Filter by Field...</option>
            {availableFilterFields.map(field => <option key={field} value={field}>{field}</option>)}
          </select>
          <select value={filterValue} onChange={(e) => setFilterValue(e.target.value)} disabled={!filterField} className="p-1 border rounded text-sm w-1/2 disabled:bg-gray-200">
            <option value="">Select Value...</option>
            {uniqueFilterValues.map(value => <option key={value} value={value}>{value}</option>)}
          </select>
        </div>
        <div className="flex gap-2 items-center text-sm">
          <label htmlFor={`aggregation-${xField}-${yField}`} className="font-medium">Group Points:</label>
          <select id={`aggregation-${xField}-${yField}`} value={aggregationMethod} onChange={(e) => setAggregationMethod(e.target.value)} className="p-1 border rounded text-sm flex-grow">
            <option value="none">No Grouping</option>
            <option value="average">By Average</option>
            <option value="sum">By Sum</option>
          </select>
        </div>
        <div className="flex items-center justify-center my-1 gap-4">
          <label className="flex items-center space-x-2 text-sm">
            <input type="checkbox" checked={showPareto} onChange={() => setShowPareto(prev => !prev)} className="accent-red-500"/>
            <span>Show Pareto Frontier</span>
          </label>
          {showPareto && (
            <select value={paretoDirection} onChange={e => setParetoDirection(e.target.value)} className="p-1 border rounded text-sm">
              <option value="top-left">Top-Left (Min X, Max Y)</option>
              <option value="top-right">Top-Right (Max X, Max Y)</option>
              <option value="bottom-left">Bottom-Left (Min X, Min Y)</option>
              <option value="bottom-right">Bottom-Right (Max X, Min Y)</option>
            </select>
          )}
        </div>
      </div>

      <div className="flex-grow flex flex-row">
        <div className="flex-grow relative">
          <Scatter data={chartData} options={chartOptions} ref={chartRef} />
        </div>
        <ScrollableLegend datasets={allDatasets} />
      </div>
      <button onClick={onExport} className="mt-2 self-end px-2 py-1 text-xs bg-gray-300 text-black rounded hover:bg-gray-400 opacity-80">
        📥 Download as PNG
      </button>

      {selectedPoint && selectedPointPosition && (
        <div
          className="absolute z-20 bg-white text-left border border-gray-300 shadow-lg rounded-md p-3 max-h-40 overflow-y-auto text-sm"
          style={{ top: `${selectedPointPosition.y}px`, left: `${selectedPointPosition.x + 15}px` }}
        >
          <div className="flex justify-end mb-1">
            <button
              onClick={onClosePopup}
              className="text-xs text-blue-500 hover:underline"
            >
              Close
            </button>
          </div>
          {Object.entries(selectedPoint).map(([key, val]) => (
            !key.startsWith('_') && <p key={key}><strong>{key}:</strong> {String(val)}</p>
          ))}
        </div>
      )}
    </div>
  );
}


// --- Main HomePage Component ---
function HomePage() {
  const [headers, setHeaders] = useState([]);
  const [dataRows, setDataRows] = useState([]);
  const [selectedFields, setSelectedFields] = useState([]);
  const [showDialog, setShowDialog] = useState(false);
  const [mode, setMode] = useState('select');
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [selectedPointPosition, setSelectedPointPosition] = useState(null);
  const [selectedPointChartIndex, setSelectedPointChartIndex] = useState(null);
  const [fullscreenIndex, setFullscreenIndex] = useState(null);
  const [fullscreenPoint, setFullscreenPoint] = useState(null);
  const [fullscreenPointPos, setFullscreenPointPos] = useState(null);
  const [colorFields, setColorFields] = useState([]);
  const [showColorDialog, setShowColorDialog] = useState(false);
  const chartRefs = useRef([]);
  const fullscreenChartContainerRef = useRef(null);

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
            setShowColorDialog(true);
          }
        });
      };
      reader.readAsText(file);
    }
  });

  const toggleColorField = (field) => {
    setColorFields(prev =>
        prev.includes(field) ? prev.filter(f => f !== field) : [...prev, field]
    );
    setSelectedFields(prev => prev.filter(f => f !== field));
  };

  const toggleField = (field) => {
    setSelectedFields(prev =>
      prev.includes(field) ? prev.filter(f => f !== field) : [...prev, field]
    );
  };

  const exportChartImage = (index) => {
    const chartInstance = chartRefs.current[index];
    if (chartInstance && chartInstance.canvas) {
      const originalCanvas = chartInstance.canvas;
      const exportCanvas = document.createElement("canvas");
      exportCanvas.width = originalCanvas.width;
      exportCanvas.height = originalCanvas.height;
      const ctx = exportCanvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
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

  const handlePointClick = (pointData, chart, element, chartIndex) => {
    setSelectedPoint(pointData);
    setSelectedPointChartIndex(chartIndex);

    const chartWrapper = chart.canvas.closest('.chart-wrapper-div');
    if (!chartWrapper) return;

    const wrapperRect = chartWrapper.getBoundingClientRect();
    const canvasRect = chart.canvas.getBoundingClientRect();

    const x = (canvasRect.left - wrapperRect.left) + element.x;
    const y = (canvasRect.top - wrapperRect.top) + element.y;

    setSelectedPointPosition({ x, y });
  };

  const colorMap = colorFields.length > 0 
    ? getColorMap(dataRows.map(row => getCompositeKey(row, colorFields))) 
    : {};

  const handleStartOver = () => {
    setMode('select');
    setSelectedFields([]);
    setColorFields([]);
    setDataRows([]);
    setHeaders([]);
    setSelectedPoint(null);
    setSelectedPointChartIndex(null);
    setSelectedPointPosition(null);
  };

  const pairs = generatePairs(selectedFields);

  return (
    <div className="flex flex-col items-center justify-between min-h-screen text-center bg-gray-100 relative">
      <header className="w-full py-8 border-b bg-white border-gray-300 shadow">
        <div className="flex items-center justify-center gap-3">
          <img src="/logo192.png" alt="logo" className="w-10 h-10" />
          <h1 className="text-4xl font-bold">PlotIt!</h1>
        </div>
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

      {showColorDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg w-[90%] max-w-md p-6 text-left">
            <h2 className="text-xl font-semibold mb-4">Which field(s) should define the point color?</h2>
            <div className="max-h-60 overflow-y-auto space-y-2">
              {headers.map((field) => (
                <label key={`color-${field}`} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={colorFields.includes(field)}
                    onChange={() => toggleColorField(field)}
                    className="accent-purple-600"
                  />
                  <span>{field}</span>
                </label>
              ))}
            </div>
            <div className="mt-6 flex justify-end">
              <button
                onClick={() => {
                  if (colorFields.length > 0) {
                    setShowColorDialog(false);
                    setShowDialog(true);
                  }
                }}
                disabled={colorFields.length === 0}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      )}

      {showDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg w-[90%] max-w-md p-6 text-left">
            <h2 className="text-xl font-semibold mb-4">What information do you want to plot?</h2>
            <div className="max-h-60 overflow-y-auto space-y-2">
              {headers.filter(h => !colorFields.includes(h)).map((field) => (
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
            <div className="mt-6 flex justify-end">
              <button
                onClick={() => {
                  setShowDialog(false);
                  setMode('plot');
                }}
                disabled={selectedFields.length < 2}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                Plot ⮕
              </button>
            </div>
          </div>
        </div>
      )}

      {mode === 'plot' && (
        <main className="flex-1 w-full p-6">
          <div className="flex justify-start mb-4">
            <button onClick={handleStartOver} className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">
              🔙 Start Over
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {pairs.map(([xField, yField], index) => (
              <ChartWrapper
                key={`${xField}-${yField}`}
                dataRows={dataRows}
                xField={xField}
                yField={yField}
                colorFields={colorFields}
                allHeaders={headers}
                colorMap={colorMap}
                onPointClick={(pointData, chart, element) => handlePointClick(pointData, chart, element, index)}
                onFullscreen={() => setFullscreenIndex(index)}
                onExport={() => exportChartImage(index)}
                chartRef={(el) => (chartRefs.current[index] = el)}
                selectedPoint={selectedPointChartIndex === index ? selectedPoint : null}
                selectedPointPosition={selectedPointChartIndex === index ? selectedPointPosition : null}
                onClosePopup={() => {
                  setSelectedPoint(null);
                  setSelectedPointChartIndex(null);
                  setSelectedPointPosition(null);
                }}
              />
            ))}
          </div>
        </main>
      )}
      
      <footer className="w-full py-6 border-t bg-white border-gray-300" />

      {fullscreenIndex !== null && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 p-4">
          <div
            ref={fullscreenChartContainerRef}
            className="relative bg-white rounded-lg shadow-lg p-6 w-full h-full flex flex-col"
          >
            <div className="flex justify-between items-center mb-4 flex-shrink-0">
              <h2 className="text-xl font-semibold">
                {pairs[fullscreenIndex][0]} vs {pairs[fullscreenIndex][1]}
              </h2>
              <button
                onClick={() => setFullscreenIndex(null)}
                className="text-gray-600 hover:text-black text-lg"
              >
                🅧
              </button>
            </div>
            
            <div className="flex-grow relative flex flex-row">
            {(() => {
              const [xField, yField] = pairs[fullscreenIndex];
              const categories = [...new Set(dataRows.map(row => getCompositeKey(row, colorFields)))];
              const datasets = categories.flatMap(category => {
                const categoryPoints = dataRows
                  .filter(row => getCompositeKey(row, colorFields) === category)
                  .map(row => {
                    const x = parseFloat(row[xField]);
                    const y = parseFloat(row[yField]);
                    return (!isNaN(x) && !isNaN(y)) ? { x, y, full: row } : null;
                  })
                  .filter(Boolean);
                  
                const scatterDataset = {
                    label: category,
                    data: categoryPoints,
                    backgroundColor: colorMap[category],
                    type: 'scatter'
                };

                return [scatterDataset];
              });

              return (
                <>
                  <div className="flex-grow h-full relative">
                    <Scatter
                      data={{ datasets }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        onClick: (event, elements, chart) => {
                          if (elements.length > 0) {
                            const { datasetIndex, index } = elements[0];
                            const pointData = chart.data.datasets[datasetIndex].data[index];
                            setFullscreenPoint(pointData.full);
                            const pointElement = elements[0];
                            const containerRect = fullscreenChartContainerRef.current.getBoundingClientRect();
                            const canvasRect = chart.canvas.getBoundingClientRect();
                            const x = pointElement.x + (canvasRect.left - containerRect.left);
                            const y = pointElement.y + (canvasRect.top - containerRect.top);
                            setFullscreenPointPos({ x, y });
                          }
                        },                    
                        plugins: {
                          legend: { display: false },
                          tooltip: {
                            callbacks: {
                              label: context => {
                                const point = context.raw;
                                const colorValue = getCompositeKey(point.full, colorFields);
                                return `${xField}: ${point.x}, ${yField}: ${point.y}` + (colorFields.length ? `, ${colorValue}` : '');
                              }
                            }
                          },
                          zoom: {
                            pan: { enabled: true, mode: 'xy' },
                            zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' },
                          },
                        },
                        scales: {
                          x: { title: { display: true, text: xField }, beginAtZero: false },
                          y: { title: { display: true, text: yField }, beginAtZero: false }
                        }
                      }}
                    />
                  </div>
                  <ScrollableLegend datasets={datasets} fullscreen={true} />
                </>
              );
            })()}
            </div>

            {fullscreenPoint && fullscreenPointPos && (
              <div
                className="absolute z-50 bg-white text-left border border-gray-300 shadow-lg rounded-md p-3 max-h-40 overflow-y-auto text-sm"
                style={{ top: fullscreenPointPos.y, left: fullscreenPointPos.x + 15 }}
              >
                <div className="flex justify-end mb-1">
                  <button
                    onClick={() => setFullscreenPoint(null)}
                    className="text-xs text-blue-500 hover:underline"
                  >
                    Close
                  </button>
                </div>
                {Object.entries(fullscreenPoint).map(([key, val]) => (
                  <p key={key}><strong>{key}:</strong> {String(val)}</p>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default HomePage;
