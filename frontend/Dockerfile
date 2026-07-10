# =============================================================================
# Frontend Dockerfile — Multi-stage build
# =============================================================================
#
# STAGE 1: BUILD
#   Uses a full Node.js image to install dependencies and build the Angular app.
#   The output is static HTML/CSS/JS files in /app/dist/frontend/browser
#
# STAGE 2: SERVE
#   Copies only the built files into a tiny Nginx image.
#   The Node.js image (600MB) is thrown away — the final image is ~25MB.
#
# Why multi-stage?
#   - Final image is tiny (~25MB vs ~650MB single-stage)
#   - No npm, Node.js, or source code in production image
#   - Faster to pull, less attack surface
#
# Build: docker build -t ai-frontend:v1 ./frontend
# Run:   docker run -p 80:80 ai-frontend:v1
# =============================================================================

# ─── STAGE 1: Build ───────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

LABEL stage=builder

WORKDIR /app

# Copy package files first (before source code).
# Docker caches each layer. If package.json hasn't changed, npm install
# is skipped on the next build — much faster CI/CD.
COPY package.json package-lock.json ./

# Install all dependencies (including devDependencies for the build)
RUN npm ci --silent

# Copy source code
COPY . .

# Build the Angular app for production.
# Output: /app/dist/frontend/browser/
# --configuration production: enables tree-shaking, minification, AOT
RUN npm run build -- --configuration production

# ─── STAGE 2: Serve ───────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS production

# Remove default Nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom Nginx config
COPY nginx.conf /etc/nginx/conf.d/app.conf

# Copy the built Angular app from the builder stage
# Only these static files go into the final image — no Node.js!
COPY --from=builder /app/dist/frontend/browser /usr/share/nginx/html

# Non-root user for security
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup && \
    chown -R appuser:appgroup /usr/share/nginx/html && \
    chown -R appuser:appgroup /var/cache/nginx && \
    chown -R appuser:appgroup /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R appuser:appgroup /var/run/nginx.pid

USER appuser

# Nginx listens on port 80
EXPOSE 80

# Nginx starts automatically as the container's main process
CMD ["nginx", "-g", "daemon off;"]
