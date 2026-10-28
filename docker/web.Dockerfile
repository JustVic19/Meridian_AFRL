FROM node:22-alpine

WORKDIR /app

COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci

COPY apps/web ./

EXPOSE 3000

CMD ["npm", "run", "dev"]
