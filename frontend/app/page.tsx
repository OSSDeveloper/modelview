import ModelViewer from './components/ModelViewer';
import ModelList from './components/ModelList';
import styles from './page.module.css';

interface Model {
  id: string;
  name: string;
  type: string;
  data: {
    nodes: Array<{ id: string; label: string; type: string }>;
    edges: Array<{ source: string; target: string }>;
  };
}

async function getModels(): Promise<Model[]> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  try {
    const res = await fetch(`${apiUrl}/api/v1/models`, {
      cache: 'no-store',
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

async function getModel(id: string): Promise<Model | null> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  try {
    const res = await fetch(`${apiUrl}/api/v1/models/${id}`, {
      cache: 'no-store',
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function Home({ searchParams }: { searchParams: { model?: string } }) {
  const models = await getModels();
  const selectedModelId = searchParams.model;
  let selectedModel: Model | null = null;
  
  if (selectedModelId) {
    selectedModel = await getModel(selectedModelId);
  }

  // Get default sample data if no model selected
  const displayNodes = selectedModel?.data?.nodes || [
    { id: '1', label: 'Start', type: 'start' },
    { id: '2', label: 'Process Task', type: 'task' },
    { id: '3', label: 'Decision', type: 'task' },
    { id: '4', label: 'End', type: 'end' },
  ];
  
  const displayEdges = selectedModel?.data?.edges || [
    { source: '1', target: '2' },
    { source: '2', target: '3' },
    { source: '3', target: '4' },
  ];

  return (
    <main className={styles.main}>
      <Head>
        <title>Model View - Process Visualization</title>
        <meta name="description" content="Interactive model visualization with D3.js" />
      </Head>

      <header className={styles.header}>
        <h1>Model View</h1>
        <p className={styles.subtitle}>Interactive Process Visualization</p>
      </header>

      <div className={styles.container}>
        <aside className={styles.sidebar}>
          <ModelList models={models} onSelect={(id) => {
            // Client-side navigation handled by link
          }} />
        </aside>

        <section className={styles.content}>
          {selectedModel && (
            <div className={styles.modelInfo}>
              <h2>{selectedModel.name}</h2>
              <span className={styles.badge}>{selectedModel.type}</span>
            </div>
          )}
          <ModelViewer nodes={displayNodes} edges={displayEdges} />
        </section>
      </div>

      <footer className={styles.footer}>
        <p>Model View Application - Powered by FastAPI & Next.js</p>
      </footer>
    </main>
  );
}