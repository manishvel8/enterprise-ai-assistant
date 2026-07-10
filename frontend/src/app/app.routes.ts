/**
 * app.routes.ts — Top-level application routes.
 *
 * Lazy loading: each feature module is only downloaded when the user navigates
 * to that route. This keeps the initial bundle small.
 *
 * Route structure:
 *   /          → redirect to /chat
 *   /chat      → Chat page (lazy loaded from ChatModule)
 *   /documents → Documents upload + library (lazy loaded from DocumentsModule)
 *   **         → redirect to /chat (catch-all for unknown URLs)
 */
import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'chat',
    pathMatch: 'full',
  },
  {
    path: 'chat',
    loadChildren: () =>
      import('./chat/chat.module').then((m) => m.ChatModule),
  },
  {
    path: 'documents',
    loadChildren: () =>
      import('./documents/documents.module').then((m) => m.DocumentsModule),
  },
  {
    path: '**',
    redirectTo: 'chat',
  },
];
