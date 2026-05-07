import { Component } from 'react';
import Head from 'next/head';

interface Model {
  id: string;
  name: string;
  type: string;
  data: {
    nodes: Array<{ id: string; label: string; type: string }>;
    edges: Array<{ source: string; target: string }>;
  };
}

interface ModelListProps {
  models: Model[];
  onSelect: (id: string) => void;
}

export default function ModelList({ models, onSelect }: ModelListProps) {
  return (
    <div className="model-list">
      <h2>Available Models</h2>
      {models.length === 0 ? (
        <p>No models available. Create one to get started.</p>
      ) : (
        <ul>
          {models.map((model) => (
            <li key={model.id} onClick={() => onSelect(model.id)}>
              <span className="model-name">{model.name}</span>
              <span className="model-type">{model.type}</span>
            </li>
          ))}
        </ul>
      )}
      <style jsx>{`
        .model-list {
          padding: 1rem;
        }
        ul {
          list-style: none;
          padding: 0;
        }
        li {
          padding: 0.75rem;
          border-bottom: 1px solid #eee;
          cursor: pointer;
          display: flex;
          justify-content: space-between;
        }
        li:hover {
          background: #f5f5f5;
        }
        .model-type {
          color: #666;
          font-size: 0.875rem;
        }
      `}</style>
    </div>
  );
}