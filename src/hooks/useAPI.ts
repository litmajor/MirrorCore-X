
import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://0.0.0.0:8000';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const fetchWithRetry = async (url: string, options?: any, maxRetries = 3) => {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await axios.get(url, { ...options, timeout: 10000 });
      return response;
    } catch (err: any) {
      const isLastRetry = i === maxRetries - 1;
      const isNetworkError = err.code === 'ECONNABORTED' || err.code === 'ERR_NETWORK';
      
      if (isLastRetry || !isNetworkError) {
        throw err;
      }
      
      // Exponential backoff: 1s, 2s, 4s
      const delay = Math.pow(2, i) * 1000;
      await sleep(delay);
    }
  }
};

export const useAPI = (endpoint: string, options?: any) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetchWithRetry(`${API_BASE_URL}${endpoint}`, options);
        setData(response?.data);
        setError(null);
      } catch (err) {
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Refresh every 5 seconds for live data
    const interval = setInterval(fetchData, 5000);
    
    return () => clearInterval(interval);
  }, [endpoint]);

  return { data, loading, error };
};
