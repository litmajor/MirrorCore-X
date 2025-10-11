
import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://0.0.0.0:8000';

export const useAPI = (endpoint: string, options?: any) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}${endpoint}`, options);
        setData(response.data);
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
