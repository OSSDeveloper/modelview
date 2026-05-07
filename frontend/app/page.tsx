'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import ModelViewer from './components/ModelViewer';
import ModelList from './components/ModelList';
import styles from './page.module.css';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface GraphNode {
  id: string;
  type: string;
  label: string;
}

interface GraphEdge {
  from: string;
  to: string;
}

interface GraphData {
  model_id: string;
  model_type: string;
  stats: Record<string, unknown>;
  nodes: GraphNode[];
  edges: GraphEdge[];
  repeated: {
    count: number;
    node_ids: string[];
    label: string;
    sub_nodes: { id: string; type: string; label: string }[];
  };
}

const POPULAR_MODELS = [
  { id: 'openai-community/gpt2', name: 'GPT-2', type: 'decoder', desc: '12-layer decoder-only' },
  { id: 'google/t5-small', name: 'T5 Small', type: 'encoder-decoder', desc: 'Small T5' },
  { id: 'openai/clip-vit-base-patch32', name: 'CLIP ViT-B/32', type: 'multimodal', desc: 'Vision-Language' },
  { id: 'microsoft/resnet-50', name: 'ResNet-50', type: 'cnn', desc: '50-layer CNN' },
  { id: 'mistralai/Mistral-7B-Instruct-v0.2', name: 'Mistral 7B', type: 'decoder', desc: 'MoE-optimized' },
  { id: 'deepseek-ai/DeepSeek-V3', name: 'DeepSeek V3', type: 'moe', desc: '128-expert MoE' },
  { id: 'meta-llama/Llama-3.1-8B-Instruct', name: 'Llama 3.1 8B', type: 'decoder', desc: '32-layer decoder' },
  { id: 'Qwen/Qwen2.5-0.8B', name: 'Qwen 0.8B', type: 'decoder', desc: 'Compact Qwen' },
];

function parseModelInput(input: string): string {
  // Strip huggingface.co prefix
  input = input.replace('https://huggingface.co/', '').replace('http://huggingface.co/', '');
  // Replace colons back to slashes (from URL routing)
  input = input.replace(':', '/');
  return input.trim();
}

async function fetchModelGraph(modelId: string): Promise<GraphData | null> {
  try {
    // modelId format: "org/model" → call /api/model/{org}/{model}
    const parts = modelId.replace(':', '/').split('/');
    if (parts.length >= 2) {
      const [org, model] = parts;
      const res = await fetch(`${API_URL}/api/model/${org}/${model}`, { cache: 'no-store' });
      if (!res.ok) return null;
      return res.json();
    }
    return null;
  } catch {
    return null;
  }
}

function ModelCard({ model, onClick }: { model: typeof POPULAR_MODELS[0]; onClick: () => void }) {
  return (
    <button className={styles.card} onClick={onClick} type="button">
      <div className={styles.cardName}>{model.name}</div>
      <div className={styles.cardMeta}>
        <span className={styles.cardType}>{model.type}</span>
        <span>{model.desc}</span>
      </div>
    </button>
  );
}

function HomeContent() {
  const searchParams = useSearchParams();
  const initialModel = searchParams.get('model') || '';

  const [inputValue, setInputValue] = useState(initialModel);
  const [modelId, setModelId] = useState(initialModel);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialModel) {
      setInputValue(initialModel);
      setModelId(initialModel);
    }
  }, [initialModel]);

  // Auto-visualize when modelId is set from URL params
  const hasAutoVisualized = useRef(false);
  useEffect(() => {
    const timer = setTimeout(() => {
      if (modelId && !hasAutoVisualized.current) {
        hasAutoVisualized.current = true;
        handleVisualize(modelId);
      }
    }, 100);
    return () => clearTimeout(timer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run once on mount

  async function handleVisualize(id?: string) {
    const targetId = id || parseModelInput(inputValue);
    if (!targetId) return;

    setLoading(true);
    setError(null);
    setModelId(targetId);
    setGraphData(null);

    const data = await fetchModelGraph(targetId);
    setLoading(false);

    if (data) {
      setGraphData(data);
    } else {
      setError(`Could not load model: ${targetId}`);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') {
      handleVisualize();
    }
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>ModelView</h1>
        <p className={styles.subtitle}>
          Visualize the architecture of any HuggingFace model
        </p>
      </header>

      <div className={styles.searchBar}>
        <input
          className={styles.searchInput}
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. openai-community/gpt2 or meta-llama/Llama-3.1-8B-Instruct"
        />
        <button className={styles.searchBtn} onClick={() => handleVisualize()} type="button">
          Visualize
        </button>
      </div>

      {!modelId && !loading && (
        <section className={styles.popular}>
          <h2 className={styles.popularTitle}>Popular Models</h2>
          <div className={styles.grid}>
            {POPULAR_MODELS.map(m => (
              <ModelCard
                key={m.id}
                model={m}
                onClick={() => {
                  setInputValue(m.id);
                  handleVisualize(m.id);
                }}
              />
            ))}
          </div>
        </section>
      )}

      {loading && (
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <span>Fetching {modelId || inputValue}...</span>
        </div>
      )}

      {error && !loading && (
        <div className={styles.error}>
          <strong>Error:</strong> {error}
          <br />
          <small>Tip: Make sure the model ID is correct and the model is public.</small>
        </div>
      )}

      {graphData && !loading && (
        <div className={styles.result}>
          <div className={styles.modelHeader}>
            <div>
              <h2 className={styles.modelName}>{graphData.model_id}</h2>
              <span className={styles.modelType}>{graphData.model_type}</span>
            </div>
            <div className={styles.stats}>
              {Object.entries(graphData.stats)
                .filter(([, v]) => v != null)
                .slice(0, 6)
                .map(([k, v]) => (
                  <span key={k} className={styles.statBadge}>
                    {k}: <strong>{String(v)}</strong>
                  </span>
                ))}
            </div>
          </div>
          <ModelViewer
            nodes={graphData.nodes}
            edges={graphData.edges}
            repeated={graphData.repeated}
          />
        </div>
      )}
    </div>
  );
}

export default function Home() {
  return (
    <Suspense fallback={<div className={styles.loading}>Loading...</div>}>
      <HomeContent />
    </Suspense>
  );
}
