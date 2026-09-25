/**
 * app.routes.ts — Top-level application routes.
 *
 * Uses loadComponent (standalone) so pages mount reliably on Angular 19.
 * NgModule loadChildren left the /chat outlet empty in the browser.
 *
 * Route structure:
 *   /          → redirect to /chat
 *   /chat      → Chat page
 *   /documents → Documents upload + library
 *   **         → redirect to /chat
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
    loadComponent: () =>
      import('./chat/chat-page/chat-page.component').then(
        (m) => m.ChatPageComponent,
      ),
  },
  {
    path: 'documents',
    loadComponent: () =>
      import('./documents/documents-page/documents-page.component').then(
        (m) => m.DocumentsPageComponent,
      ),
  },
  {
    path: '**',
    redirectTo: 'chat',
  },
];
