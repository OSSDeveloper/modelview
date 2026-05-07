FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine
# Next.js standalone output is in .next/standalone/
COPY --from=0 /app/.next/standalone /usr/share/nginx/html
COPY --from=0 /app/.next/static /usr/share/nginx/html/.next/static
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
