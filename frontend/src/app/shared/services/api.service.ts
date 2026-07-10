/**
 * api.service.ts — Central service for all HTTP calls to the FastAPI backend.
 *
 * Why a single ApiService?
 *   - One place to set the base URL (from environment)
 *   - One place to add auth headers later
 *   - One place to handle common errors (401 Unauthorized, 500 Server Error)
 *   - Components never import HttpClient directly — they use this service
 *
 * Pattern: Injectable service that wraps HttpClient with typed responses.
 */

import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { environment } from '../../../environments/environment';
import {
  ChatRequest,
  ChatResponse,
  ChatHistoryResponse,
} from '../models/chat.model';
import {
  UploadResponse,
  DocumentMetadata,
  DocumentListResponse,
} from '../models/document.model';

@Injectable({
  providedIn: 'root',   // Singleton: one instance shared across the whole app
})
export class ApiService {
  private baseUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  // ─────────────────────────────────────────────────────────────────
  // Chat API
  // ─────────────────────────────────────────────────────────────────

  /**
   * Send a chat message to the backend.
   * Milestone 2: Returns echo response.
   * Milestone 4+: Returns OpenAI-generated response.
   */
  sendMessage(request: ChatRequest): Observable<ChatResponse> {
    return this.http
      .post<ChatResponse>(`${this.baseUrl}/api/chat`, request)
      .pipe(catchError(this.handleError));
  }

  /**
   * Get conversation history for a session.
   */
  getChatHistory(sessionId: string): Observable<ChatHistoryResponse> {
    return this.http
      .get<ChatHistoryResponse>(`${this.baseUrl}/api/chat/history/${sessionId}`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Clear conversation history for a session.
   */
  clearChatHistory(sessionId: string): Observable<void> {
    return this.http
      .delete<void>(`${this.baseUrl}/api/chat/history/${sessionId}`)
      .pipe(catchError(this.handleError));
  }

  // ─────────────────────────────────────────────────────────────────
  // Documents API
  // ─────────────────────────────────────────────────────────────────

  /**
   * Upload a document file.
   * Uses FormData because we're sending a binary file.
   */
  uploadDocument(file: File): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http
      .post<UploadResponse>(`${this.baseUrl}/api/documents/upload`, formData)
      .pipe(catchError(this.handleError));
  }

  /**
   * Get all uploaded documents.
   */
  getDocuments(): Observable<DocumentListResponse> {
    return this.http
      .get<DocumentListResponse>(`${this.baseUrl}/api/documents`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Get status of a specific document.
   * Poll this after uploading to know when processing is complete.
   */
  getDocument(documentId: string): Observable<DocumentMetadata> {
    return this.http
      .get<DocumentMetadata>(`${this.baseUrl}/api/documents/${documentId}`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Delete a document.
   */
  deleteDocument(documentId: string): Observable<void> {
    return this.http
      .delete<void>(`${this.baseUrl}/api/documents/${documentId}`)
      .pipe(catchError(this.handleError));
  }

  // ─────────────────────────────────────────────────────────────────
  // Health
  // ─────────────────────────────────────────────────────────────────

  checkHealth(): Observable<{ status: string; version: string }> {
    return this.http
      .get<{ status: string; version: string }>(`${this.baseUrl}/health`)
      .pipe(catchError(this.handleError));
  }

  // ─────────────────────────────────────────────────────────────────
  // Error Handling
  // ─────────────────────────────────────────────────────────────────

  private handleError(error: HttpErrorResponse): Observable<never> {
    let message = 'An unexpected error occurred.';
    if (error.status === 0) {
      // Network error — backend not reachable
      message = 'Cannot reach the server. Is it running?';
    } else if (error.status === 400) {
      message = error.error?.detail || 'Bad request.';
    } else if (error.status === 404) {
      message = error.error?.detail || 'Not found.';
    } else if (error.status === 413) {
      message = 'File is too large.';
    } else if (error.status === 422) {
      message = 'Validation error: ' + JSON.stringify(error.error?.detail);
    } else if (error.status === 500) {
      message = 'Server error. Check backend logs.';
    }
    console.error('[ApiService]', error.status, message);
    return throwError(() => new Error(message));
  }
}
