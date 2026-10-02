# syntax=docker/dockerfile:1

FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build && npm prune --omit=dev

FROM node:24-alpine
ENV NODE_ENV=production
WORKDIR /app
COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY config.example.json ./config.json
EXPOSE 8080
# Note: engine start/stop commands (and VRAM checks) run INSIDE this container.
# For engines on the host, set "managed": false on each model in config.json
# and point targetBaseUrl at the host (host.docker.internal on Docker).
CMD ["node", "dist/cli.js", "--config", "config.json"]
