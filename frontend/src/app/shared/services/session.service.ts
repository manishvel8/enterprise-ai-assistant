/**
 * session.service.ts — Manages user identity and session IDs.
 *
 * Why a session?
 *   Each conversation is a "session". The session_id links messages together
 *   so the backend can return conversation history and maintain context.
 *
 *   Session IDs are generated in the browser and stored in localStorage so
 *   they survive page refreshes.
 */

import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class SessionService {
  private readonly USER_ID_KEY = 'ai_user_id';
  private readonly SESSION_ID_KEY = 'ai_session_id';
  private readonly SESSIONS_KEY = 'ai_sessions';

  private sessionsSubject: BehaviorSubject<string[]>;
  sessions$;

  private currentSessionSubject: BehaviorSubject<string>;
  currentSession$;

  constructor() {
    // Initialize subjects carefully — never call .next on a subject that
    // is still being constructed (that blanked the Chat page).
    const sessions = this.loadSessions();
    let current = localStorage.getItem(this.SESSION_ID_KEY);

    if (!current) {
      current = 'session_' + this.generateId();
      localStorage.setItem(this.SESSION_ID_KEY, current);
    }

    if (!sessions.includes(current)) {
      sessions.unshift(current);
      this.saveSessions(sessions.slice(0, 20));
    }

    this.sessionsSubject = new BehaviorSubject<string[]>(sessions.slice(0, 20));
    this.sessions$ = this.sessionsSubject.asObservable();

    this.currentSessionSubject = new BehaviorSubject<string>(current);
    this.currentSession$ = this.currentSessionSubject.asObservable();
  }

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
    const sessions = [sessionId, ...this.sessionsSubject.value].slice(0, 20);
    this.sessionsSubject.next(sessions);
    this.saveSessions(sessions);
    this.currentSessionSubject.next(sessionId);
    localStorage.setItem(this.SESSION_ID_KEY, sessionId);
    return sessionId;
  }

  /** Switch to an existing session. */
  switchSession(sessionId: string): void {
    this.currentSessionSubject.next(sessionId);
    localStorage.setItem(this.SESSION_ID_KEY, sessionId);
  }

  private loadSessions(): string[] {
    try {
      const raw = localStorage.getItem(this.SESSIONS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  private saveSessions(sessions: string[]): void {
    localStorage.setItem(this.SESSIONS_KEY, JSON.stringify(sessions));
  }

  private generateId(): string {
    return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  }
}
