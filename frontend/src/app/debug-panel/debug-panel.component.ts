/**
 * debug-panel.component.ts — Debug panel showing agentic workflow internals.
 *
 * Milestone 29: Shows real-time observability data from each chat request:
 *   - Intent classification (Router Agent output)
 *   - Retrieved chunks and similarity scores (Retriever Agent output)
 *   - Cypher query and graph results (Cypher Agent output)
 *   - Token usage and cost (Answer Agent output)
 *   - Latency breakdown (overall workflow)
 *   - Critic validation result
 *   - Langfuse trace ID (link to observability dashboard)
 *
 * This is invaluable for:
 *   - Debugging RAG quality (are the right chunks being retrieved?)
 *   - Monitoring costs (which queries are expensive?)
 *   - Understanding routing decisions (why was this a graph_question?)
 *   - Interview demonstrations ("here's what the AI is actually doing")
 */

import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface DebugData {
  intent?: string;
  rewrittenQuery?: string;
  retrievedChunksCount?: number;
  similarityScores?: number[];
  cypherQuery?: string;
  cypherResultsCount?: number;
  tokenUsage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  costUsd?: number;
  latencyMs?: number;
  langfuseTraceId?: string;
  isGrounded?: boolean;
  criticIterations?: number;
  errors?: string[];
}

@Component({
  selector: 'app-debug-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="debug-panel" *ngIf="data">
      <div class="debug-header">
        <span class="debug-title">🔍 Debug Panel</span>
        <span class="debug-latency" *ngIf="data.latencyMs">
          {{ data.latencyMs | number:'1.0-0' }}ms
        </span>
      </div>

      <!-- Intent -->
      <div class="debug-section" *ngIf="data.intent">
        <div class="debug-label">Intent</div>
        <div class="debug-value intent-badge" [class]="'intent-' + data.intent">
          {{ data.intent }}
        </div>
      </div>

      <!-- Rewritten Query -->
      <div class="debug-section" *ngIf="data.rewrittenQuery">
        <div class="debug-label">Rewritten Query</div>
        <div class="debug-value debug-code">{{ data.rewrittenQuery }}</div>
      </div>

      <!-- Retrieved Chunks -->
      <div class="debug-section" *ngIf="data.retrievedChunksCount !== undefined">
        <div class="debug-label">Retrieved Chunks</div>
        <div class="debug-value">{{ data.retrievedChunksCount }} chunks</div>
        <div class="similarity-bars" *ngIf="data.similarityScores?.length">
          <div class="sim-label">Similarity Scores:</div>
          <div class="sim-bar-container" *ngFor="let score of data.similarityScores; let i = index">
            <span class="sim-idx">{{ i + 1 }}</span>
            <div class="sim-bar">
              <div class="sim-fill" [style.width]="(score * 100) + '%'"></div>
            </div>
            <span class="sim-score">{{ score | number:'1.2-2' }}</span>
          </div>
        </div>
      </div>

      <!-- Cypher Query -->
      <div class="debug-section" *ngIf="data.cypherQuery">
        <div class="debug-label">Cypher Query</div>
        <div class="debug-value debug-code cypher">{{ data.cypherQuery }}</div>
        <div class="debug-value small" *ngIf="data.cypherResultsCount !== undefined">
          {{ data.cypherResultsCount }} graph results
        </div>
      </div>

      <!-- Token Usage -->
      <div class="debug-section" *ngIf="data.tokenUsage">
        <div class="debug-label">Token Usage</div>
        <div class="token-grid">
          <div class="token-item">
            <span class="token-label">Prompt</span>
            <span class="token-value">{{ data.tokenUsage.prompt_tokens | number }}</span>
          </div>
          <div class="token-item">
            <span class="token-label">Completion</span>
            <span class="token-value">{{ data.tokenUsage.completion_tokens | number }}</span>
          </div>
          <div class="token-item">
            <span class="token-label">Total</span>
            <span class="token-value">{{ data.tokenUsage.total_tokens | number }}</span>
          </div>
          <div class="token-item" *ngIf="data.costUsd !== undefined">
            <span class="token-label">Cost</span>
            <span class="token-value cost">{{ (data.costUsd || 0) | currency:'USD':'symbol':'1.4-4' }}</span>
          </div>
        </div>
      </div>

      <!-- Critic Validation -->
      <div class="debug-section" *ngIf="data.isGrounded !== undefined">
        <div class="debug-label">Grounding Check</div>
        <div class="debug-value" [class.grounded]="data.isGrounded" [class.not-grounded]="!data.isGrounded">
          {{ data.isGrounded ? '✅ Grounded' : '⚠️ Not Grounded' }}
          <span *ngIf="data.criticIterations && data.criticIterations > 1">
            ({{ data.criticIterations }} iterations)
          </span>
        </div>
      </div>

      <!-- Langfuse Trace -->
      <div class="debug-section" *ngIf="data.langfuseTraceId">
        <div class="debug-label">Langfuse Trace</div>
        <div class="debug-value">
          <a [href]="langfuseUrl + data.langfuseTraceId" target="_blank" class="trace-link">
            {{ data.langfuseTraceId?.slice(0, 12) }}... →
          </a>
        </div>
      </div>

      <!-- Errors -->
      <div class="debug-section errors" *ngIf="data.errors?.length">
        <div class="debug-label">Errors</div>
        <div class="error-item" *ngFor="let error of data.errors">⚠️ {{ error }}</div>
      </div>
    </div>

    <div class="debug-empty" *ngIf="!data">
      <p>Debug information will appear here after your first message.</p>
    </div>
  `,
  styles: [`
    .debug-panel {
      font-family: 'JetBrains Mono', 'Consolas', monospace;
      font-size: 12px;
      color: #e5e7eb;
      padding: 12px;
      height: 100%;
      overflow-y: auto;
    }
    .debug-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #374151;
    }
    .debug-title { font-weight: bold; font-size: 13px; }
    .debug-latency { color: #60a5fa; }
    .debug-section {
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid #1f2937;
    }
    .debug-label {
      color: #9ca3af;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }
    .debug-value { color: #d1d5db; }
    .debug-code {
      background: #0f172a;
      padding: 6px 8px;
      border-radius: 4px;
      white-space: pre-wrap;
      word-break: break-all;
      font-size: 11px;
      max-height: 80px;
      overflow-y: auto;
    }
    .cypher { color: #7dd3fc; max-height: 120px; }
    .small { font-size: 11px; color: #6b7280; margin-top: 4px; }
    .intent-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
    }
    .intent-doc_question { background: #1e3a5f; color: #60a5fa; }
    .intent-graph_question { background: #1f3124; color: #4ade80; }
    .intent-general_chat { background: #292222; color: #f87171; }
    .intent-summary { background: #2d2419; color: #fbbf24; }
    .similarity-bars { margin-top: 6px; }
    .sim-label { color: #6b7280; font-size: 10px; margin-bottom: 4px; }
    .sim-bar-container {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 3px;
    }
    .sim-idx { width: 12px; color: #6b7280; }
    .sim-bar {
      flex: 1;
      height: 6px;
      background: #1f2937;
      border-radius: 3px;
      overflow: hidden;
    }
    .sim-fill {
      height: 100%;
      background: #3b82f6;
      border-radius: 3px;
      transition: width 0.3s ease;
    }
    .sim-score { width: 32px; color: #9ca3af; }
    .token-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .token-item {
      display: flex;
      justify-content: space-between;
      background: #1f2937;
      padding: 4px 8px;
      border-radius: 4px;
    }
    .token-label { color: #6b7280; }
    .token-value { color: #d1d5db; }
    .cost { color: #fbbf24; }
    .grounded { color: #4ade80; }
    .not-grounded { color: #fbbf24; }
    .trace-link { color: #60a5fa; text-decoration: underline; cursor: pointer; }
    .errors .error-item { color: #f87171; margin-top: 4px; }
    .debug-empty {
      color: #4b5563;
      font-size: 12px;
      padding: 12px;
      text-align: center;
    }
  `]
})
export class DebugPanelComponent implements OnChanges {
  @Input() data: DebugData | null = null;

  langfuseUrl = 'https://cloud.langfuse.com/trace/';

  ngOnChanges() {
    // Component re-renders automatically when @Input changes
  }
}
