/**
 * document.model.ts — TypeScript interfaces for document upload and library.
 */

export type DocumentStatus = 'pending' | 'processing' | 'complete' | 'failed';

export interface UploadResponse {
  document_id: string;
  file_name: string;
  file_type: string;
  status: DocumentStatus;
  message: string;
}

export interface DocumentMetadata {
  document_id: string;
  file_name: string;
  file_type: string;
  file_size_bytes: number;
  status: DocumentStatus;
  chunk_count: number;
  error_message?: string;
  created_at?: string;
  updated_at?: string;
}

export interface DocumentListResponse {
  documents: DocumentMetadata[];
  total: number;
}
