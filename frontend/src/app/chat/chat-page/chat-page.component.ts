/**
 * chat-page.component.ts — Main chat screen.
 *
 * Layout:
 *   ┌─────────────────────────────────────────────────────────┐
 *   │  Sidebar (history)  │   Chat area          │  Debug     │
 *   │  - New Chat button  │   - Message list     │  panel     │
 *   │  - Session list     │   - Input box        │  (toggle)  │
 *   └─────────────────────────────────────────────────────────┘
 *
 * State:
 *   - messages: array of {role, content, citations, debug} for the current session
 *   - isLoading: true while waiting for backend response
 *   - inputText: what the user is typing
 *
 * Data flow:
 *   User types → presses Enter → sendMessage() → ApiService.sendMessage()
 *   → backend returns ChatResponse → push assistant message → scroll to bottom
 */

import {
  Component,
  OnInit,
  OnDestroy,
  ViewChild,
  ElementRef,
  ChangeDetectorRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

import { ApiService } from '../../shared/services/api.service';
import { SessionService } from '../../shared/services/session.service';
import { ChatRequest, ChatResponse, SourceCitation, DebugInfo } from '../../shared/models/chat.model';

/** One displayed message in the chat window. */
interface DisplayMessage {
  role: 'user' | 'assistant';
  content: string;
  citations?: SourceCitation[];
  debug?: DebugInfo;
  isError?: boolean;
  timestamp: Date;
}

@Component({
  selector: 'app-chat-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat-page.component.html',
  styleUrl: './chat-page.component.scss',
})
export class ChatPageComponent implements OnInit, OnDestroy {
  @ViewChild('messageListEl') messageListEl!: ElementRef<HTMLDivElement>;
  @ViewChild('inputEl') inputEl!: ElementRef<HTMLTextAreaElement>;

  messages: DisplayMessage[] = [];
  inputText = '';
  isLoading = false;
  showDebugPanel = false;
  showSidebar = true;

  sessions: string[] = [];
  currentSessionId = '';

  private destroy$ = new Subject<void>();

  constructor(
    private api: ApiService,
    public sessionService: SessionService,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    // Subscribe to session list for the sidebar
    this.sessionService.sessions$
      .pipe(takeUntil(this.destroy$))
      .subscribe((sessions) => {
        this.sessions = sessions;
      });

    // Subscribe to current session changes
    this.sessionService.currentSession$
      .pipe(takeUntil(this.destroy$))
      .subscribe((sessionId) => {
        if (sessionId !== this.currentSessionId) {
          this.currentSessionId = sessionId;
          this.loadHistory();
        }
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /** Load conversation history for the current session from the backend. */
  loadHistory(): void {
    if (!this.currentSessionId) return;
    this.messages = [];

    this.api
      .getChatHistory(this.currentSessionId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (resp) => {
          this.messages = resp.messages.map((m) => ({
            role: m.role as 'user' | 'assistant',
            content: m.content,
            timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
          }));
          this.scrollToBottom();
        },
        error: () => {
          // History not found is fine — it's a new session
          this.messages = [];
        },
      });
  }

  /** Start a brand-new conversation. */
  newChat(): void {
    this.sessionService.newSession();
    this.messages = [];
    this.inputText = '';
    setTimeout(() => this.inputEl?.nativeElement?.focus(), 100);
  }

  /** Switch to a different session from the sidebar. */
  switchSession(sessionId: string): void {
    this.sessionService.switchSession(sessionId);
  }

  /** Send the user's message to the backend. */
  sendMessage(): void {
    const text = this.inputText.trim();
    if (!text || this.isLoading) return;

    // Add user message immediately to the UI
    this.messages.push({
      role: 'user',
      content: text,
      timestamp: new Date(),
    });

    this.inputText = '';
    this.isLoading = true;
    this.scrollToBottom();

    const request: ChatRequest = {
      user_id: this.sessionService.userId,
      session_id: this.currentSessionId,
      message: text,
      document_ids: [],
      stream: false,           // Streaming added in Milestone 24
    };

    this.api
      .sendMessage(request)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (resp: ChatResponse) => {
          this.messages.push({
            role: 'assistant',
            content: resp.answer,
            citations: resp.citations,
            debug: resp.debug,
            timestamp: new Date(),
          });
          this.isLoading = false;
          this.scrollToBottom();
        },
        error: (err: Error) => {
          this.messages.push({
            role: 'assistant',
            content: `Error: ${err.message}`,
            isError: true,
            timestamp: new Date(),
          });
          this.isLoading = false;
          this.scrollToBottom();
        },
      });
  }

  /** Allow pressing Enter to send (Shift+Enter for new line). */
  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  /** Scroll the message list to the bottom after adding a new message. */
  private scrollToBottom(): void {
    setTimeout(() => {
      const el = this.messageListEl?.nativeElement;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    }, 50);
  }

  /** Format session ID to a human-readable label for the sidebar. */
  sessionLabel(sessionId: string): string {
    const shortId = sessionId.slice(-8);
    return `Session ${shortId}`;
  }

  /** Auto-grow the textarea as the user types. */
  autoGrow(event: Event): void {
    const el = event.target as HTMLTextAreaElement;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }

  toggleDebugPanel(): void {
    this.showDebugPanel = !this.showDebugPanel;
  }

  toggleSidebar(): void {
    this.showSidebar = !this.showSidebar;
  }
}
