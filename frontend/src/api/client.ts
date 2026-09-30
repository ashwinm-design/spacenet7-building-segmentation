/**
 * frontend/src/api/client.ts
 *
 * Configurable Axios client for FastAPI backend communication.
 */

import axios from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180000, // 3 minutes timeout for heavy spatial tiles
  headers: {
    'Accept': 'application/json',
  },
});

export default apiClient;
