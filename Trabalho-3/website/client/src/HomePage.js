import React from 'react';
import { useDropzone } from 'react-dropzone';

function HomePage() {
  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      'text/csv': ['.csv']
    },
    onDrop: acceptedFiles => {
      console.log(acceptedFiles);
      // Aqui você pode fazer o upload ou leitura do CSV
    }
  });

  return (
    <div className="flex flex-col items-center justify-between min-h-screen px-4 py-8 text-center bg-gray-50">
      <header className="w-full py-4 border-b border-gray-300">
        <h1 className="text-3xl font-bold">PlotIt</h1>
      </header>

      <main className="flex flex-col items-center justify-center flex-grow">
        <p className="mb-4 text-lg">Place your CSV and we will PlotIt!</p>
        <div
          {...getRootProps()}
          className="p-4 border border-dashed border-gray-400 rounded-md cursor-pointer hover:bg-gray-100"
        >
          <input {...getInputProps()} />
          <p>Choose your csv file</p>
        </div>
      </main>

      <footer className="w-full py-4 border-t border-gray-300"></footer>
    </div>
  );
}

export default HomePage;
