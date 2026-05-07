'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface Node {
  id: string;
  label: string;
  type: string;
  x?: number;
  y?: number;
}

interface Edge {
  source: string;
  target: string;
}

interface ModelViewerProps {
  nodes: Node[];
  edges: Edge[];
  width?: number;
  height?: number;
}

export default function ModelViewer({ nodes, edges, width = 800, height = 600 }: ModelViewerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    try {
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();

      // Create zoom behavior
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
          g.attr('transform', event.transform);
        });

      svg.call(zoom);

      const g = svg.append('g');

      // Create force simulation
      const simulation = d3.forceSimulation(nodes as d3.SimulationNodeDatum[])
        .force('link', d3.forceLink(edges).id((d: any) => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(40));

      // Draw edges
      const link = g.append('g')
        .selectAll('line')
        .data(edges)
        .join('line')
        .attr('stroke', '#999')
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.6);

      // Draw nodes
      const node = g.append('g')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .call(d3.drag<SVGGElement, Node>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }) as any);

      node.append('circle')
        .attr('r', 20)
        .attr('fill', (d) => d.type === 'start' ? '#4CAF50' : d.type === 'end' ? '#f44336' : '#2196F3');

      node.append('text')
        .text((d) => d.label)
        .attr('text-anchor', 'middle')
        .attr('dy', 35)
        .attr('font-size', '12px');

      // Update positions on tick
      simulation.on('tick', () => {
        link
          .attr('x1', (d: any) => d.source.x)
          .attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x)
          .attr('y2', (d: any) => d.target.y);

        node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
      });

    } catch (err) {
      setError('Failed to render visualization');
    }
  }, [nodes, edges, width, height]);

  if (nodes.length === 0) {
    return (
      <div className="empty-state">
        <p>No model data to display</p>
        <style jsx>{`
          .empty-state {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 400px;
            color: #666;
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className="model-viewer">
      <svg ref={svgRef} width={width} height={height} />
      {error && <p className="error">{error}</p>}
      <style jsx>{`
        .model-viewer {
          border: 1px solid #ddd;
          border-radius: 8px;
          overflow: hidden;
        }
        svg {
          display: block;
          background: #fafafa;
        }
        .error {
          color: red;
          padding: 1rem;
        }
      `}</style>
    </div>
  );
}