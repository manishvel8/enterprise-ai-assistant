/**
 * app.component.ts — Root application component.
 *
 * This is the shell that wraps every page.
 * It renders the top navigation bar and the <router-outlet> where
 * page components (ChatPage, DocumentsPage) are loaded based on the route.
 *
 * Think of this as the outermost HTML shell of the application.
 */
import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'Enterprise AI Assistant';
  version = '1.0.0 · Milestone 2';
}
