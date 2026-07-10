/**
 * chat.model.ts — TypeScript interfaces that mirror the backend Pydantic schemas.
 *
 * Why mirror the backend?
 * Angular is strongly typed. By defining these interfaces here, the compiler
 * catches any mismatch between what the API returns and what the UI expects.
 */

export type MessageRole = 'user' | 'assistant' | 'system';

export interface Message {
  role: MessageRole;
  content: string;
  timestamp?: Date;
}

export interface ChatRequest {
  user_id: string;
  session_id: string;
  message: string;
  document_ids?: string[];
  stream?: boolean;
}

export interface SourceCitation {
  chunk_id: string;
  document_id: string;
  file_name: string;
  page_number?: number;
  section_title?: string;
  excerpt: string;
}

export interface DebugInfo {
  intent?: string;
  rewritten_query?: string;
  retrieved_chunks_count: number;
  similarity_scores: number[];
  cypher_query?: string;
  cypher_results_count: number;
  token_usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  cost_usd?: number;
  latency_ms?: number;
  langfuse_trace_id?: string;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  citations: SourceCitation[];
  debug?: DebugInfo;
}

export interface ChatHistoryItem {
  session_id: string;
  role: MessageRole;
  content: string;
  timestamp?: string;
}

export interface ChatHistoryResponse {
  session_id: string;
  messages: ChatHistoryItem[];
}
