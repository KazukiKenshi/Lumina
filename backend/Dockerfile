FROM node:18-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --silent || npm install --silent

COPY server.js .
COPY drop-username-index.js .
COPY auth.js .
COPY models/ ./models/
COPY config ./config/

EXPOSE 5000

CMD ["npm", "start"]