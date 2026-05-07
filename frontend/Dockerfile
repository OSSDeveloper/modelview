FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Extract standalone server for minimal image
FROM node:20-alpine
WORKDIR /app

COPY --from=0 /app/.next/standalone /app
COPY --from=0 /app/.next/static /app/.next/static
COPY --from=0 /app/public /app/public 2>/dev/null || true

ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

CMD ["node", "server.js"]
