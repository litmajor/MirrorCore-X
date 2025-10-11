
import { useState, useEffect, useCallback } from 'react';

export const useWebSocket = (url: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [data, setData] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.type === 'initial_state') {
          setData(message.data);
        } else if (message.type === 'oracle_update') {
          setData((prev: any) => ({ 
            ...prev, 
            oracle_directives: message.data 
          }));
        } else if (message.type === 'scanner_update') {
          setData((prev: any) => ({ 
            ...prev, 
            scanner_data: message.data 
          }));
        } else {
          setData((prev: any) => ({ ...prev, ...message.data }));
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setSocket(ws);

    return () => {
      ws.close();
    };
  }, [url]);

  const sendCommand = useCallback((command: string, params?: any) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify({ command, params }));
    }
  }, [socket, isConnected]);

  return { data, isConnected, sendCommand };
};
