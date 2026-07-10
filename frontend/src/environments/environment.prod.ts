/**
 * environment.prod.ts — Production environment configuration.
 * Used when running: ng build --configuration production
 */
export const environment = {
  production: true,
  apiUrl: '',                // Empty string = same origin (Nginx proxies /api/*)
  enableDebugPanel: false,   // Disable debug panel in production
  appVersion: '1.0.0',
};
