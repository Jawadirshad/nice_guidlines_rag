# client/Dockerfile

# Step 1: Build the React App
FROM node:18 AS build

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json first (to leverage Docker cache)
COPY client/package.json client/package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the app
COPY client/ ./

# Build the Vite app
RUN npm run build

# Step 2: Serve the React App using Nginx
FROM nginx:alpine

# Copy the built files from the build stage to Nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80 for Nginx to serve the app
EXPOSE 80

# Run Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
