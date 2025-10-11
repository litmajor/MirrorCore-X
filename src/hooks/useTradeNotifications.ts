import { useEffect } from 'react';
import { toast } from 'sonner';
import { useWebSocket } from './useWebSocket';

interface Trade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  timestamp: string;
  pnl?: number;
}

export const useTradeNotifications = () => {
  const { data: wsData } = useWebSocket('ws://0.0.0.0:8000/ws');

  useEffect(() => {
    if (wsData?.new_trade) {
      const trade: Trade = wsData.new_trade;
      
      const isProfitable = trade.pnl !== undefined && trade.pnl > 0;
      const toastFn = isProfitable ? toast.success : toast.error;
      
      const title = `${trade.side} ${trade.symbol}`;
      const description = `${trade.quantity} @ $${trade.price.toFixed(2)}${
        trade.pnl !== undefined ? ` | P&L: $${trade.pnl.toFixed(2)}` : ''
      }`;
      
      toastFn(title, {
        description,
        duration: 5000,
      });
    }
  }, [wsData?.new_trade]);

  return {
    showTradeNotification: (trade: Trade) => {
      const isProfitable = trade.pnl !== undefined && trade.pnl > 0;
      const toastFn = isProfitable ? toast.success : toast.error;
      
      const title = `${trade.side} ${trade.symbol}`;
      const description = `${trade.quantity} @ $${trade.price.toFixed(2)}${
        trade.pnl !== undefined ? ` | P&L: $${trade.pnl.toFixed(2)}` : ''
      }`;
      
      toastFn(title, {
        description,
        duration: 5000,
      });
    }
  };
};
