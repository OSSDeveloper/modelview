'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface Node {
  id: string;
  type: string;
  label: string;
}

interface Edge {
  from: string;
  to: string;
}

interface RepeatedData {
  count: number;
  node_ids: string[];
  label: string;
  sub_nodes: { id: string; type: string; label: string }[];
}

interface ModelViewerProps {
  nodes: Node[];
  edges: Edge[];
  repeated?: RepeatedData;
  width?: number;
  height?: number;
}

const TYPE_COLORS: Record<string, string> = {
  embedding: '#4ade80',
  positional_embedding: '#86efac',
  layer_norm: '#a78bfa',
  attention: '#60a5fa',
  mlp: '#fb923c',
  router: '#f472b6',
  expert: '#f87171',
  decoder_block: '#818cf8',
  encoder_block: '#34d399',
  vision_block: '#c084fc',
  moe_block: '#f472b6',
  lm_head: '#fbbf24',
  mlp_head: '#a3e635',
  projection: '#22d3ee',
  vision_encoder: '#c084fc',
  text_encoder: '#60a5fa',
  contrastive_head: '#f472b6',
  vision: '#c084fc',
  text: '#60a5fa',
  conv: '#4ade80',
  batch_norm: '#a78bfa',
  pool: '#86efac',
  residual_block: '#818cf8',
  fc: '#fbbf24',
  patch_embed: '#22d3ee',
  class_token: '#fb923c',
  concat: '#f9a8d4',
};

function getColor(type: string): string {
  return TYPE_COLORS[type] || '#6366f1';
}

export default function ModelViewer({
  nodes,
  edges,
  repeated,
  width = 900,
  height = 600,
}: ModelViewerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    try {
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();

      // Zoom behavior
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
          g.attr('transform', event.transform);
        });
      svg.call(zoom);

      const g = svg.append('g');

      // Normalize edges: support both {from,to} and {source,target}
      const normalizedEdges = edges.map(e => ({
        source: (e as any).from || (e as any).source,
        target: (e as any).to || (e as any).target,
      }));

      // Node lookup map
      const nodeMap = new Map(nodes.map(n => [n.id, n]));

      // D3 simulation — use a layered layout instead of pure force
      // Layer assignment: walk the graph to determine layers
      const nodeLayer = new Map<string, number>();
      const visited = new Set<string>();

      const assignLayer = (nodeId: string, layer: number) => {
        if (visited.has(nodeId)) return;
        visited.add(nodeId);
        nodeLayer.set(nodeId, layer);
        // Find outgoing edges
        normalizedEdges.forEach(e => {
          if (e.source === nodeId) {
            assignLayer(e.target as string, layer + 1);
          }
        });
      };

      // Find roots (nodes with no incoming edges)
      const hasIncoming = new Set(normalizedEdges.map(e => e.target as string));
      nodes.forEach(n => {
        if (!hasIncoming.has(n.id)) {
          assignLayer(n.id, 0);
        }
      });

      // Assign any unvisited nodes
      nodes.forEach(n => {
        if (!visited.has(n.id)) {
          assignLayer(n.id, 0);
        }
      });

      const maxLayer = Math.max(...Array.from(nodeLayer.values()), 0);
      const layerGroups: Map<number, string[]> = new Map();
      nodeLayer.forEach((layer, nodeId) => {
        if (!layerGroups.has(layer)) layerGroups.set(layer, []);
        layerGroups.get(layer)!.push(nodeId);
      });

      // Position nodes by layer
      const positionedNodes = nodes.map(n => {
        const layer = nodeLayer.get(n.id) || 0;
        const siblings = layerGroups.get(layer) || [];
        const idx = siblings.indexOf(n.id);
        const layerWidth = width - 100;
        const x = 50 + (layer / Math.max(maxLayer, 1)) * layerWidth;
        const y = 60 + ((idx + 0.5) / siblings.length) * (height - 120);
        return { ...n, x, y };
      });

      const posMap = new Map(positionedNodes.map(n => [n.id, n]));

      // Draw edges as curved paths
      const linkGen = d3.linkHorizontal<any, any>()
        .source(d => [d.source.x, d.source.y])
        .target(d => [d.target.x, d.target.y]);

      const link = g.append('g')
        .selectAll('path')
        .data(normalizedEdges)
        .join('path')
        .attr('fill', 'none')
        .attr('stroke', '#374151')
        .attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.7)
        .attr('d', d => {
          const s = posMap.get(d.source as string);
          const t = posMap.get(d.target as string);
          if (!s || !t) return '';
          return linkGen({ source: s, target: t } as any);
        });

      // Arrow markers
      svg.append('defs').append('marker')
        .attr('id', 'arrow')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#4b5563');

      link.attr('marker-end', 'url(#arrow)');

      // Draw repeated block indicator
      if (repeated && repeated.count > 0) {
        const repNodes = positionedNodes.filter(n => n.id.startsWith('block_') || n.id.startsWith('vit_block_'));
        if (repNodes.length > 0) {
          const minX = Math.min(...repNodes.map(n => n.x));
          const maxX = Math.max(...repNodes.map(n => n.x));
          const minY = Math.min(...repNodes.map(n => n.y));
          const maxY = Math.max(...repNodes.map(n => n.y));

          g.append('rect')
            .attr('x', minX - 15)
            .attr('y', minY - 20)
            .attr('width', maxX - minX + 30)
            .attr('height', maxY - minY + 40)
            .attr('fill', 'rgba(129, 140, 248, 0.06)')
            .attr('stroke', '#6366f1')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '4,4')
            .attr('rx', 6);

          g.append('text')
            .attr('x', minX - 20)
            .attr('y', minY - 28)
            .attr('fill', '#818cf8')
            .attr('font-size', '11px')
            .attr('font-weight', '600')
            .text(`× ${repeated.count}`);
        }
      }

      // Draw nodes
      const nodeGs = g.append('g')
        .selectAll('g')
        .data(positionedNodes)
        .join('g')
        .style('cursor', 'default');

      // Node rectangles
      nodeGs.append('rect')
        .attr('x', d => d.x - 60)
        .attr('y', d => d.y - 18)
        .attr('width', 120)
        .attr('height', 36)
        .attr('rx', 6)
        .attr('fill', d => getColor(d.type))
        .attr('fill-opacity', 0.15)
        .attr('stroke', d => getColor(d.type))
        .attr('stroke-width', 1.5);

      // Node label
      nodeGs.append('text')
        .attr('x', d => d.x)
        .attr('y', d => d.y + 4)
        .attr('text-anchor', 'middle')
        .attr('fill', '#e5e7eb')
        .attr('font-size', '10px')
        .attr('font-weight', '500')
        .text(d => {
          const label = d.label;
          return label.length > 18 ? label.substring(0, 16) + '…' : label;
        });

      // Type badge
      nodeGs.append('text')
        .attr('x', d => d.x + 52)
        .attr('y', d => d.y - 8)
        .attr('text-anchor', 'end')
        .attr('fill', d => getColor(d.type))
        .attr('font-size', '8px')
        .attr('font-family', 'monospace')
        .text(d => d.type);

      // Drag behavior
      nodeGs.call(d3.drag<SVGGElement, typeof positionedNodes[0]>()
        .on('start', (event, d) => {
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
          d.x = event.x;
          d.y = event.y;
          link.attr('d', (e: any) => {
            const s = posMap.get(e.source as string) || e.source;
            const t = posMap.get(e.target as string) || e.target;
            if (s && t) {
              return linkGen({ source: s, target: t } as any);
            }
            return '';
          });
        })
        .on('end', (_, d) => {
          d.fx = null;
          d.fy = null;
        }) as any);

    } catch (err) {
      setError('Failed to render visualization');
    }
  }, [nodes, edges, repeated, width, height]);

  if (nodes.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 400, color: '#666' }}>
        No graph data to display
      </div>
    );
  }

  return (
    <div style={{ position: 'relative' }}>
      {error && (
        <div style={{ color: '#f87171', padding: '12px', textAlign: 'center' }}>
          {error}
        </div>
      )}
      <svg
        ref={svgRef}
        width="100%"
        height={height}
        style={{ background: '#0f0f1a', borderRadius: '10px', border: '1px solid #1e1e2e' }}
      />
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 12, justifyContent: 'center' }}>
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '11px', color: '#888' }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: color }} />
            <span style={{ fontFamily: 'monospace' }}>{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
