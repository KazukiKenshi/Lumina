import React, { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Link } from 'react-router-dom';
import axios from 'axios';
import './ChatHistoryPage.css';

// Paginated date list; click a date to view full sessions for that date.
const PAGE_SIZE = 7;

const ChatHistoryPage = () => {
  const { token } = useAuth();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [grouped, setGrouped] = useState({});
  const [dailyDominant, setDailyDominant] = useState({});
  const [dates, setDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState(null);
  const [page, setPage] = useState(0);
  const API_URL = process.env.REACT_APP_BACKEND_URL || '';

  useEffect(() => {
    const fetchHistory = async () => {
      setLoading(true); setError(null);
      try {
        const headers = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const resp = await axios.get(`${API_URL}/api/chat-history?limit=400&skip=0`, { headers });
        if (!resp.data.success) throw new Error('Failed to load history');
        const histories = resp.data.chatHistories || [];
        const byDate = {};
        histories.forEach(h => {
          const dateKey = new Date(h.createdAt).toISOString().split('T')[0];
          if (!byDate[dateKey]) byDate[dateKey] = [];
          byDate[dateKey].push(h);
        });
        // Compute dominant user emotion per date
        const dominantMap = {};
        Object.entries(byDate).forEach(([date, sessions]) => {
          const counts = { happy:0, sad:0, neutral:0, angry:0, anxious:0, calm:0 };
          sessions.forEach(s => {
            s.messages.forEach(m => {
              if (m.role === 'user' && counts[m.emotion] !== undefined) {
                counts[m.emotion]++;
              }
            });
          });
          // Determine dominant
          let dom = 'neutral';
          let max = -1;
            Object.entries(counts).forEach(([emo,c]) => { if (c > max) { max = c; dom = emo; } });
          dominantMap[date] = dom;
        });
        const sortedDates = Object.keys(byDate).sort((a,b) => b.localeCompare(a));
        setGrouped(byDate);
        setDates(sortedDates);
        if (sortedDates.length) setSelectedDate(sortedDates[0]);
        setDailyDominant(dominantMap);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [token]);

  const formatTime = iso => { try { return new Date(iso).toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' }); } catch { return iso; } };

  const pagedDates = dates.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(dates.length / PAGE_SIZE);

  const emojiForEmotion = (emo) => {
    switch(emo){
      case 'happy': return '😀';
      case 'sad': return '😢';
      case 'angry': return '😠';
      case 'anxious': return '😰';
      case 'calm': return '😌';
      case 'neutral':
      default: return '😐';
    }
  };

  return (
    <div style={{ minHeight:'100vh', background:'linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #2b6cb0 100%)', color:'#fff', fontFamily:'system-ui, sans-serif' }}>
      <div style={{ padding:'20px 30px', borderBottom:'1px solid rgba(255,255,255,0.15)', background:'rgba(26,54,93,0.85)', backdropFilter:'blur(10px)', position:'sticky', top:0, zIndex:50, display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <h1 style={{ margin:0, fontSize:'1.75rem', letterSpacing:'-0.5px' }}>Chat History</h1>
        <div style={{ display:'flex', gap:'12px' }}>
          <Link to="/" className="nav-link" style={{ textDecoration:'none', padding:'10px 20px', borderRadius:'8px', border:'1px solid rgba(255,255,255,0.3)', color:'#fff' }}>Home</Link>
          <Link to="/play" className="nav-link" style={{ textDecoration:'none', padding:'10px 20px', borderRadius:'8px', border:'1px solid rgba(255,255,255,0.3)', color:'#fff' }}>Lumina</Link>
        </div>
      </div>
      <div style={{ display:'flex', gap:'28px', padding:'30px', maxWidth:'1400px', margin:'0 auto' }}>
        {/* Date list panel */}
        <div style={{ width:'250px', flexShrink:0, position:'sticky', top:'110px', alignSelf:'flex-start' }}>
          <h2 style={{ fontSize:'1.1rem', margin:'0 0 12px' }}>Dates</h2>
          {loading && <div style={{ fontSize:'0.9rem' }}>Loading...</div>}
          {error && <div style={{ color:'#ff6b6b', fontSize:'0.85rem' }}>Error: {error}</div>}
          {!loading && !error && dates.length === 0 && <div style={{ fontSize:'0.85rem' }}>No history yet.</div>}
          <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
            {pagedDates.map(d => (
              <button key={d} onClick={() => setSelectedDate(d)} style={{
                textAlign:'left',
                padding:'10px 12px',
                borderRadius:'8px',
                border:'1px solid rgba(255,255,255,0.25)',
                background: d === selectedDate ? 'rgba(79,209,197,0.25)' : 'rgba(255,255,255,0.08)',
                color:'#fff',
                cursor:'pointer',
                fontSize:'0.85rem',
                fontWeight:500,
                letterSpacing:'0.3px'
              }}>{emojiForEmotion(dailyDominant[d])} {d}</button>
            ))}
          </div>
          {/* Pagination controls */}
          {totalPages > 1 && (
            <div style={{ display:'flex', justifyContent:'space-between', marginTop:'14px' }}>
              <button disabled={page===0} onClick={() => setPage(p => Math.max(0, p-1))} style={{
                padding:'6px 10px', borderRadius:'6px', border:'1px solid rgba(255,255,255,0.3)', background:'rgba(255,255,255,0.1)', color:'#fff', cursor: page===0 ? 'not-allowed':'pointer', fontSize:'0.75rem'
              }}>Prev</button>
              <div style={{ fontSize:'0.75rem', opacity:0.8, alignSelf:'center' }}>{page+1}/{totalPages}</div>
              <button disabled={page===totalPages-1} onClick={() => setPage(p => Math.min(totalPages-1, p+1))} style={{
                padding:'6px 10px', borderRadius:'6px', border:'1px solid rgba(255,255,255,0.3)', background:'rgba(255,255,255,0.1)', color:'#fff', cursor: page===totalPages-1 ? 'not-allowed':'pointer', fontSize:'0.75rem'
              }}>Next</button>
            </div>
          )}
        </div>
        {/* Sessions display */}
        <div style={{ flex:1 }}>
          {selectedDate && grouped[selectedDate] && (
            <div>
              <h2 style={{ fontSize:'1.4rem', margin:'0 0 18px', letterSpacing:'-0.5px' }}>{selectedDate}</h2>
              {grouped[selectedDate].map(session => (
                <div key={session._id} style={{
                  background:'rgba(255,255,255,0.08)',
                  backdropFilter:'blur(4px)',
                  border:'1px solid rgba(255,255,255,0.15)',
                  borderRadius:'14px',
                  padding:'16px 18px',
                  marginBottom:'18px',
                  boxShadow:'0 4px 16px rgba(0,0,0,0.25)'
                }}>
                  {/* <div style={{ fontSize:'11px', opacity:0.7, marginBottom:'10px' }}>Session: {session._id}</div> */}
                  <div style={{ display:'flex', flexDirection:'column', gap:'10px' }}>
                    {session.messages.map(m => {
                      const isUser = m.role === 'user';
                      return (
                        <div key={m._id || m.timestamp} className={`history-message-row ${isUser ? 'user' : 'assistant'}`}>
                          {isUser ? (
                            <>
                              <div className="history-bubble user">{m.content}</div>
                              <span className="history-timestamp user-ts">{formatTime(m.timestamp)}</span>
                            </>
                          ) : (
                            <>
                              <div className="history-bubble assistant">{m.content}</div>
                              <span className="history-timestamp assistant-ts">{formatTime(m.timestamp)}</span>
                            </>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
          {!selectedDate && !loading && <div style={{ fontSize:'0.9rem' }}>Select a date to view sessions.</div>}
        </div>
      </div>
    </div>
  );
};

export default ChatHistoryPage;
