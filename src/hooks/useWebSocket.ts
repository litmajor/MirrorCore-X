
import { useState, useEffect, useCallback, useRef } from 'react';

export const useWebSocket = (url: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [data, setData] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const connect = useCallback(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setIsConnected(true);
      reconnectAttemptsRef.current = 0;
      console.log('WebSocket connected');
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
      
      // Exponential backoff reconnection: 1s, 2s, 4s, 8s, max 30s
      const delay = Math.min(Math.pow(2, reconnectAttemptsRef.current) * 1000, 30000);
      reconnectAttemptsRef.current++;
      
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log(`Reconnecting... (attempt ${reconnectAttemptsRef.current})`);
        connect();
      }, delay);
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
  }, [url]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      socket?.close();
    };
  }, [connect]);

  const sendCommand = useCallback((command: string, params?: any) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify({ command, params }));
    }
  }, [socket, isConnected]);

  return { data, isConnected, sendCommand };
};
