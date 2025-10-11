
import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, Send, Sparkles, TrendingUp, AlertCircle, Lightbulb } from 'lucide-react';
import axios from 'axios';

interface ChatMessage {
  query: string;
  response: any;
  timestamp: string;
}

const AIInsights: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showProactiveInsights, setShowProactiveInsights] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  useEffect(() => {
    // Load conversation history
    axios.get('http://0.0.0.0:8000/api/ai/conversation-history')
      .then(res => setMessages(res.data.history || []))
      .catch(err => console.error('Failed to load history:', err));
  }, []);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    setLoading(true);
    try {
      const response = await axios.post('http://0.0.0.0:8000/api/ai/chat', {
        query: input
      });

      setMessages(prev => [...prev, response.data]);
      setInput('');
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setLoading(false);
    }
  };

  const quickQuestions = [
    "How is my performance?",
    "What are the top signals?",
    "Explain current market conditions",
    "What's the risk level?",
    "Show top strategies"
  ];

  const renderResponse = (response: any) => {
    if (!response) return null;

    return (
      <div className="space-y-3">
        <p className="text-white font-medium">{response.summary}</p>
        
        {response.interpretation && (
          <p className="text-txt-secondary">{response.interpretation}</p>
        )}

        {response.strategy_breakdown && (
          <pre className="text-sm text-txt-secondary bg-bg-tertiary p-3 rounded">
            {response.strategy_breakdown}
          </pre>
        )}

        {response.signal_breakdown && (
          <pre className="text-sm text-txt-secondary bg-bg-tertiary p-3 rounded">
            {response.signal_breakdown}
          </pre>
        )}

        {response.recommendation && (
          <div className="flex items-start space-x-2 bg-brand-cyan/10 p-3 rounded border border-brand-cyan/30">
            <Lightbulb className="w-5 h-5 text-brand-cyan flex-shrink-0 mt-0.5" />
            <p className="text-brand-cyan text-sm">{response.recommendation}</p>
          </div>
        )}

        {response.recommendations && response.recommendations.length > 0 && (
          <div className="space-y-2">
            <p className="text-sm font-semibold text-white">Recommendations:</p>
            <ul className="list-disc list-inside space-y-1 text-sm text-txt-secondary">
              {response.recommendations.map((rec: string, idx: number) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </div>
        )}

        {response.topics && (
          <div className="space-y-1">
            {response.topics.map((topic: string, idx: number) => (
              <p key={idx} className="text-sm text-txt-secondary">{topic}</p>
            ))}
          </div>
        )}

        {response.top_opportunity && (
          <div className="bg-bg-surface p-3 rounded border border-accent-dark">
            <p className="text-sm font-semibold text-white mb-2">Top Opportunity:</p>
            <div className="space-y-1 text-sm">
              <p className="text-brand-cyan">{response.top_opportunity.symbol} - {response.top_opportunity.action}</p>
              <p className="text-txt-secondary">Confidence: {response.top_opportunity.score}%</p>
              <p className="text-txt-secondary">{response.top_opportunity.reason}</p>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="h-[calc(100vh-80px)] flex flex-col">
      <div className="page-header mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Sparkles className="w-8 h-8 text-brand-purple" />
            <div>
              <h1 className="page-title">Mirrax AI</h1>
              <p className="page-subtitle">Your MirrorCore-X Intelligence Assistant</p>
            </div>
          </div>
          <div className="text-sm text-txt-secondary">
            <span className="px-3 py-1 bg-brand-cyan/20 border border-brand-cyan/30 rounded-full">
              Multi-Agent Cognitive System
            </span>
          </div>
        </div>
      </div>

      {/* Quick Questions */}
      {messages.length === 0 && (
        <div className="mb-4">
          <p className="text-txt-secondary text-sm mb-3">Quick Questions:</p>
          <div className="flex flex-wrap gap-2">
            {quickQuestions.map((q, idx) => (
              <button
                key={idx}
                onClick={() => setInput(q)}
                className="px-3 py-2 text-sm bg-bg-surface hover:bg-accent-dark text-txt-secondary hover:text-white rounded-lg border border-accent-dark transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
        {messages.map((msg, idx) => (
          <div key={idx} className="space-y-3">
            {/* User Query */}
            <div className="flex justify-end">
              <div className="bg-brand-cyan/20 border border-brand-cyan/30 rounded-lg px-4 py-2 max-w-2xl">
                <p className="text-white">{msg.query}</p>
              </div>
            </div>

            {/* AI Response */}
            <div className="flex justify-start">
              <div className="bg-bg-surface border border-accent-dark rounded-lg px-4 py-3 max-w-3xl">
                {renderResponse(msg.response)}
                <div className="flex items-center justify-between mt-2 pt-2 border-t border-accent-dark">
                  <p className="text-xs text-txt-secondary">
                    {msg.response.assistant || 'Mirrax AI'} • {new Date(msg.timestamp).toLocaleTimeString()}
                  </p>
                  {msg.response.confidence && (
                    <span className="text-xs text-brand-cyan">
                      {(msg.response.confidence * 100).toFixed(0)}% confident
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-accent-dark pt-4">
        <div className="flex space-x-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about performance, strategies, signals, or market conditions..."
            className="flex-1 bg-bg-surface border border-accent-dark rounded-lg px-4 py-3 text-white placeholder-txt-secondary focus:outline-none focus:border-brand-cyan"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-6 py-3 bg-gradient-to-r from-brand-cyan to-brand-purple text-white rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-brand-cyan/30 transition-all"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AIInsights;
