/**
 * session.service.ts — Manages user identity and session IDs.
 *
 * Why a session?
 *   Each conversation is a "session". The session_id links messages together
 *   so the backend can return conversation history and maintain context.
 *
 *   In this milestone: session IDs are generated in the browser and stored in
 *   localStorage so they survive page refreshes.
 *
 *   Later milestones will add proper user authentication and server-side sessions.
 */

import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class SessionService {
  private readonly USER_ID_KEY = 'ai_user_id';
  private readonly SESSION_ID_KEY = 'ai_session_id';

  /** Observable list of all session IDs for the history sidebar. */
  private sessionsSubject = new BehaviorSubject<string[]>(this.loadSessions());
  sessions$ = this.sessionsSubject.asObservable();

  /** The currently active session ID. */
  private currentSessionSubject = new BehaviorSubject<string>(this.getOrCreateSession());
  currentSession$ = this.currentSessionSubject.asObservable();

  get userId(): string {
    let id = localStorage.getItem(this.USER_ID_KEY);
    if (!id) {
      id = 'user_' + this.generateId();
      localStorage.setItem(this.USER_ID_KEY, id);
    }
    return id;
  }

  get currentSessionId(): string {
    return this.currentSessionSubject.value;
  }

  /** Start a new conversation session. */
  newSession(): string {
    const sessionId = 'session_' + this.generateId();
    this.currentSessionSubject.next(sessionId);
    const sessions = this.sessionsSubject.value;
    if (!sessions.includes(sessionId)) {
      const updated = [sessionId, ...sessions].slice(0, 20); // keep last 20
      this.sessionsSubject.next(updated);
      this.saveSessions(updated);
    }
    localStorage.setItem(this.SESSION_ID_KEY, sessionId);
    return sessionId;
  }

  /** Switch to an existing session. */
  switchSession(sessionId: string): void {
    this.currentSessionSubject.next(sessionId);
    localStorage.setItem(this.SESSION_ID_KEY, sessionId);
  }

  private getOrCreateSession(): string {
    const existing = localStorage.getItem(this.SESSION_ID_KEY);
    if (existing) {
      const sessions = this.sessionsSubject.value;
      if (!sessions.includes(existing)) {
        const updated = [existing, ...sessions].slice(0, 20);
        this.sessionsSubject.next(updated);
        this.saveSessions(updated);
      }
      return existing;
    }
    return this.newSession();
  }

  private loadSessions(): string[] {
    try {
      const raw = localStorage.getItem('ai_sessions');
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  private saveSessions(sessions: string[]): void {
    localStorage.setItem('ai_sessions', JSON.stringify(sessions));
  }

  private generateId(): string {
    return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  }
}
