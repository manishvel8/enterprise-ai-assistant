/**
 * environment.ts — Development environment configuration.
 *
 * This file is replaced by environment.prod.ts during `ng build --configuration production`.
 * Angular's file replacement is configured in angular.json.
 *
 * Why separate environment files?
 *   - Development: API runs on localhost:8000, debug enabled
 *   - Production: API served from same domain via Nginx proxy (/api/*), debug disabled
 */
export const environment = {
  production: false,

  /**
   * Base URL for all API calls.
   * In development: Angular dev server proxies /api/* to http://localhost:8000
   * In production (Nginx): /api/* is proxied to the backend service
   */
  apiUrl: 'http://localhost:8000',

  /**
   * Enable the debug panel that shows:
   * - Intent classification
   * - Retrieved chunks with similarity scores
   * - Cypher query
   * - Token usage, cost, latency
   * - Langfuse trace ID
   */
  enableDebugPanel: true,

  /**
   * App version — shown in the UI footer.
   * Bumped manually per milestone.
   */
  appVersion: '1.0.0-m2',
};
