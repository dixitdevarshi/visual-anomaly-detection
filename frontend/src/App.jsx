import { useState } from 'react'
import Landing from './pages/Landing'
import Detector from './pages/Detector'

export default function App() {
  const [page, setPage] = useState('landing')

  return (
    <div style={{ backgroundColor: '#0D1117', minHeight: '100vh', color: '#E6EDF3', fontFamily: 'Inter, sans-serif' }}>
      {page === 'landing' ? (
        <Landing onStart={() => setPage('detector')} />
      ) : (
        <Detector onBack={() => setPage('landing')} />
      )}
    </div>
  )
}
