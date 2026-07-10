/**
 * app.config.ts — Root application providers.
 *
 * In Angular 17+ standalone API, this replaces AppModule for top-level configuration.
 * Providers here are available throughout the entire application.
 */
import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';

import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes),
    /**
     * provideHttpClient — registers Angular's HttpClient for dependency injection.
     * Required for ApiService to inject HttpClient.
     * withInterceptorsFromDi() — allows adding HTTP interceptors later (for auth headers).
     */
    provideHttpClient(withInterceptorsFromDi()),
  ],
};
