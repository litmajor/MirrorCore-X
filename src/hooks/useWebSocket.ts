
import { useState, useEffect, useCallback } from 'react';
import io, { Socket } from 'socket.io-client';

export const useWebSocket = (url: string) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [data, setData] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const socketInstance = io(url, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    socketInstance.on('connect', () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    });

    socketInstance.on('disconnect', () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    });

    socketInstance.on('dashboard_update', (newData) => {
      setData(newData);
    });

    setSocket(socketInstance);

    return () => {
      socketInstance.disconnect();
    };
  }, [url]);

  const sendCommand = useCallback((command: string, params?: any) => {
    if (socket && isConnected) {
      socket.emit(command, params);
    }
  }, [socket, isConnected]);

  return { data, isConnected, sendCommand };
};
