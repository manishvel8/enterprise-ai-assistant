/**
 * documents-page.component.ts — Document upload + library screen.
 *
 * Layout:
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  Upload Zone (drag & drop or click to browse)                    │
 *   │  - Shows selected files with file type badges                    │
 *   │  - Upload progress bar per file                                  │
 *   └──────────────────────────────────────────────────────────────────┘
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  Document Library                                                │
 *   │  - Table: file name | type | size | status | actions            │
 *   │  - Status badge: pending | processing | complete | failed        │
 *   │  - Refresh button to poll status                                 │
 *   └──────────────────────────────────────────────────────────────────┘
 */

import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, interval } from 'rxjs';
import { takeUntil, switchMap, startWith } from 'rxjs/operators';

import { ApiService } from '../../shared/services/api.service';
import { DocumentMetadata, DocumentStatus } from '../../shared/models/document.model';

interface UploadItem {
  file: File;
  status: 'pending' | 'uploading' | 'done' | 'error';
  progress: number;
  documentId?: string;
  error?: string;
}

@Component({
  selector: 'app-documents-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './documents-page.component.html',
  styleUrl: './documents-page.component.scss',
})
export class DocumentsPageComponent implements OnInit, OnDestroy {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  documents: DocumentMetadata[] = [];
  uploadQueue: UploadItem[] = [];
  isDragOver = false;
  isLoadingDocuments = false;
  errorMessage = '';

  readonly allowedExtensions = [
    'pdf', 'docx', 'pptx', 'xlsx', 'csv', 'txt',
    'png', 'jpg', 'jpeg', 'mp3', 'mp4', 'wav',
  ];

  /** Pre-computed accept string for the file input element. */
  readonly acceptString = this.allowedExtensions.map((e) => '.' + e).join(',');

  private destroy$ = new Subject<void>();

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.loadDocuments();
    // Poll document status every 5 seconds for any 'processing' documents
    interval(5000)
      .pipe(
        startWith(0),
        takeUntil(this.destroy$),
      )
      .subscribe(() => {
        const hasProcessing = this.documents.some(
          (d) => d.status === 'pending' || d.status === 'processing',
        );
        if (hasProcessing) {
          this.loadDocuments();
        }
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /** Load the document list from the backend. */
  loadDocuments(): void {
    this.api
      .getDocuments()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (resp) => {
          this.documents = resp.documents;
          this.isLoadingDocuments = false;
        },
        error: (err) => {
          console.error('Failed to load documents:', err);
          this.isLoadingDocuments = false;
        },
      });
  }

  /** Open the file picker dialog. */
  openFilePicker(): void {
    this.fileInput.nativeElement.click();
  }

  /** Handle files selected via the file picker. */
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      this.addFilesToQueue(Array.from(input.files));
      input.value = '';  // reset so the same file can be re-selected
    }
  }

  /** Handle files dropped onto the drop zone. */
  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = false;
    const files = event.dataTransfer?.files;
    if (files) {
      this.addFilesToQueue(Array.from(files));
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = true;
  }

  onDragLeave(event: DragEvent): void {
    this.isDragOver = false;
  }

  /** Validate and add files to the upload queue. */
  addFilesToQueue(files: File[]): void {
    for (const file of files) {
      const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
      if (!this.allowedExtensions.includes(ext)) {
        this.errorMessage = `File type '${ext}' is not supported.`;
        setTimeout(() => (this.errorMessage = ''), 5000);
        continue;
      }
      this.uploadQueue.push({
        file,
        status: 'pending',
        progress: 0,
      });
    }
    // Auto-start uploads
    this.uploadAll();
  }

  /** Upload all pending files in the queue. */
  uploadAll(): void {
    const pending = this.uploadQueue.filter((item) => item.status === 'pending');
    for (const item of pending) {
      this.uploadFile(item);
    }
  }

  /** Upload a single file. */
  uploadFile(item: UploadItem): void {
    item.status = 'uploading';
    item.progress = 10;

    this.api
      .uploadDocument(item.file)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (resp) => {
          item.status = 'done';
          item.progress = 100;
          item.documentId = resp.document_id;
          this.loadDocuments();
          // Auto-remove from queue after 3 seconds
          setTimeout(() => {
            this.uploadQueue = this.uploadQueue.filter((q) => q !== item);
          }, 3000);
        },
        error: (err) => {
          item.status = 'error';
          item.progress = 0;
          item.error = err.message;
        },
      });

    // Simulate progress while waiting for upload
    const progressInterval = setInterval(() => {
      if (item.status === 'uploading' && item.progress < 90) {
        item.progress += Math.random() * 15;
      } else {
        clearInterval(progressInterval);
      }
    }, 300);
  }

  /** Remove an item from the upload queue. */
  removeFromQueue(item: UploadItem): void {
    this.uploadQueue = this.uploadQueue.filter((q) => q !== item);
  }

  /** Delete a document from the library. */
  deleteDocument(document_id: string): void {
    if (!confirm('Delete this document? This cannot be undone.')) return;

    this.api
      .deleteDocument(document_id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.documents = this.documents.filter((d) => d.document_id !== document_id);
        },
        error: (err) => {
          this.errorMessage = `Failed to delete: ${err.message}`;
        },
      });
  }

  /** Format bytes into human-readable size string. */
  formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  /** Get CSS class for the document status badge. */
  statusClass(status: DocumentStatus): string {
    return {
      pending: 'badge--warning',
      processing: 'badge--warning',
      complete: 'badge--success',
      failed: 'badge--error',
    }[status] ?? '';
  }

  /** True if any document is still being processed (used by the polling notice). */
  get hasProcessingDocuments(): boolean {
    return this.documents.some(
      (d) => d.status === 'pending' || d.status === 'processing',
    );
  }

  /** Get emoji icon for file type. */
  fileTypeIcon(fileType: string): string {
    const icons: Record<string, string> = {
      pdf: '📄',
      docx: '📝',
      pptx: '📊',
      xlsx: '📈',
      csv: '📋',
      txt: '📃',
      png: '🖼️',
      jpg: '🖼️',
      jpeg: '🖼️',
      mp3: '🎵',
      mp4: '🎥',
      wav: '🎵',
    };
    return icons[fileType.toLowerCase()] ?? '📁';
  }
}
