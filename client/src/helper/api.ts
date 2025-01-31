import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';

// Load environment variables (for Node.js environments)
if (process.env.NODE_ENV !== 'production') {
  require('dotenv').config();
}

// Set up your base URL from the environment variable
const api = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || 'https://localhost:3000', // Fallback URL if not set
  headers: {
    'Content-Type': 'application/json',
  },
});

// GET request
export const get = async <T>(endpoint: string, config?: AxiosRequestConfig): Promise<T> => {
  try {
    const response: AxiosResponse<T> = await api.get(endpoint, config);
    return response.data;
  } catch (error) {
    throw new Error(`GET request failed: ${error}`);
  }
};

// POST request
export const post = async <T>(endpoint: string, data: any, config?: AxiosRequestConfig): Promise<T> => {
  try {
    const response: AxiosResponse<T> = await api.post(endpoint, data, config);
    return response.data;
  } catch (error) {
    throw new Error(`POST request failed: ${error}`);
  }
};

// PUT request
export const put = async <T>(endpoint: string, data: any, config?: AxiosRequestConfig): Promise<T> => {
  try {
    const response: AxiosResponse<T> = await api.put(endpoint, data, config);
    return response.data;
  } catch (error) {
    throw new Error(`PUT request failed: ${error}`);
  }
};

// DELETE request
export const del = async <T>(endpoint: string, config?: AxiosRequestConfig): Promise<T> => {
  try {
    const response: AxiosResponse<T> = await api.delete(endpoint, config);
    return response.data;
  } catch (error) {
    throw new Error(`DELETE request failed: ${error}`);
  }
};
