import { useState } from 'react'
import { useDetection } from '../hooks/useDetection'

const CATEGORIES = [
  'bottle', 'cable', 'capsule', 'carpet', 'grid',
  'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
  'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

const THRESHOLD = 30

export default function Detector({ onBack }) {
  const [imageFile, setImageFile] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [category, setCategory] = useState('bottle')
  const [isDragging, setIsDragging] = useState(false)
  const [activeView, setActiveView] = useState('overlay')
  const { result, loading, error, detect, reset } = useDetection()

  const handleFile = (file) => {
  if (!file || !file.type.startsWith('image/')) return
  setImageFile(file)
  setImageUrl(URL.createObjectURL(file))
  reset()
  setActiveView('overlay')
  setTimeout(() => {
    document.getElementById('category-select')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, 100)
}

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handleReset = () => {
    setImageFile(null)
    setImageUrl(null)
    reset()
  }

  const isAnomaly = result ? result.score > THRESHOLD : false

  const views = {
    original: result?.original,
    heatmap: result?.heatmap,
    overlay: result?.overlay,
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* Nav */}
      <nav style={{
        padding: '16px 48px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid #21262D',
        position: 'sticky',
        top: 0,
        background: '#0D1117',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <button
            onClick={onBack}
            style={{
              background: 'none',
              border: '1px solid #30363D',
              color: '#8B949E',
              padding: '6px 12px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '13px',
              fontFamily: 'Inter, sans-serif',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            ← Back
          </button>
          <span style={{ fontWeight: 600, fontSize: '15px' }}>Detection Workspace</span>
        </div>
        {result && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 14px',
            borderRadius: '20px',
            border: `1px solid ${isAnomaly ? 'rgba(248,81,73,0.3)' : 'rgba(63,185,80,0.3)'}`,
            background: isAnomaly ? 'rgba(248,81,73,0.05)' : 'rgba(63,185,80,0.05)',
          }}>
            <div style={{
              width: '6px', height: '6px', borderRadius: '50%',
              background: isAnomaly ? '#F85149' : '#3FB950'
            }} />
            <span style={{
              fontSize: '13px', fontWeight: 500,
              color: isAnomaly ? '#F85149' : '#3FB950'
            }}>
              {isAnomaly ? 'Anomaly Detected' : 'Normal'}
            </span>
            <span style={{ fontSize: '12px', color: '#8B949E', fontFamily: 'monospace' }}>
              {result.score.toFixed(2)}
            </span>
          </div>
        )}
      </nav>

      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '380px 1fr',
        gap: '0',
      }}>

        {/* Left sidebar */}
        <div style={{
          borderRight: '1px solid #21262D',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px'
        }}>

          {/* Upload area */}
          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onClick={() => document.getElementById('file-input').click()}
            style={{
              border: `2px dashed ${isDragging ? '#58A6FF' : '#30363D'}`,
              borderRadius: '10px',
              padding: '32px 20px',
              textAlign: 'center',
              cursor: 'pointer',
              background: isDragging ? 'rgba(88,166,255,0.04)' : 'transparent',
              transition: 'all 0.2s'
            }}
          >
            <input
              id="file-input"
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
            {imageUrl ? (
              <img
                src={imageUrl}
                alt="preview"
                style={{
                  width: '100%',
                  maxHeight: '200px',
                  objectFit: 'contain',
                  borderRadius: '6px'
                }}
              />
            ) : (
              <>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
                  stroke="#8B949E" strokeWidth="1.5" style={{ marginBottom: '12px' }}>
                  <path strokeLinecap="round" strokeLinejoin="round"
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
                <p style={{ fontSize: '14px', color: '#E6EDF3', marginBottom: '4px' }}>
                  Drop image here
                </p>
                <p style={{ fontSize: '12px', color: '#8B949E' }}>
                  or click to browse
                </p>
              </>
            )}
          </div>

          {imageUrl && (
            <button
              onClick={() => document.getElementById('file-input').click()}
              style={{
                background: 'none',
                border: '1px solid #30363D',
                color: '#8B949E',
                padding: '8px',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '13px',
                fontFamily: 'Inter, sans-serif',
                marginTop: '-10px'
              }}
            >
              Change image
            </button>
          )}

          {/* Category */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <label style={{ fontSize: '12px', color: '#8B949E' }}>
              Product category
            </label>
            <select
              id="category-select"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              disabled={loading}
              style={{
                background: '#161B22',
                border: '1px solid #30363D',
                color: '#E6EDF3',
                padding: '10px 12px',
                borderRadius: '6px',
                fontSize: '14px',
                cursor: 'pointer',
                fontFamily: 'Inter, sans-serif',
                outline: 'none'
              }}
            >
              {CATEGORIES.map(cat => (
                <option key={cat} value={cat}>
                  {cat.replace('_', ' ')}
                </option>
              ))}
            </select>
            <p style={{ fontSize: '11px', color: '#484F58' }}>
              Select the category that matches your uploaded image
            </p>
          </div>

          {/* Actions */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: 'auto' }}>
            <button
              onClick={() => detect(imageFile, category)}
              disabled={!imageFile || loading}
              style={{
                padding: '12px',
                borderRadius: '8px',
                background: !imageFile || loading ? '#21262D' : '#58A6FF',
                color: !imageFile || loading ? '#484F58' : '#0D1117',
                border: 'none',
                fontSize: '14px',
                fontWeight: 600,
                cursor: !imageFile || loading ? 'not-allowed' : 'pointer',
                fontFamily: 'Inter, sans-serif',
                transition: 'all 0.2s'
              }}
            >
              {loading ? 'Running detection...' : 'Run Detection'}
            </button>
            {(imageFile || result) && (
              <button
                onClick={handleReset}
                disabled={loading}
                style={{
                  padding: '10px',
                  borderRadius: '8px',
                  background: 'transparent',
                  color: '#8B949E',
                  border: '1px solid #30363D',
                  fontSize: '13px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontFamily: 'Inter, sans-serif'
                }}
              >
                Reset
              </button>
            )}
          </div>

          {/* Model info */}
          <div style={{
            padding: '16px',
            borderRadius: '8px',
            background: '#161B22',
            border: '1px solid #21262D'
          }}>
            <p style={{ fontSize: '11px', color: '#484F58', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '1px' }}>
              Model
            </p>
            {[
              ['Backbone', 'DINOv2 ViT-B/14'],
              ['Algorithm', 'PatchCore'],
              ['Patches', '256 per image'],
              ['Features', '768 dims'],
              ['AUROC', '0.9781'],
            ].map(([k, v]) => (
              <div key={k} style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: '6px'
              }}>
                <span style={{ fontSize: '12px', color: '#8B949E' }}>{k}</span>
                <span style={{ fontSize: '12px', color: '#E6EDF3', fontFamily: 'monospace' }}>{v}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right — results */}
        <div style={{ padding: '32px 40px', display: 'flex', flexDirection: 'column', gap: '32px' }}>

          {!imageFile && !loading && (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '16px',
              color: '#484F58',
              textAlign: 'center'
            }}>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <div>
                <p style={{ fontSize: '16px', color: '#8B949E', marginBottom: '8px' }}>
                  Upload an image to get started
                </p>
                <p style={{ fontSize: '13px' }}>
                  Results will appear here after detection runs
                </p>
              </div>
            </div>
          )}

          {imageFile && !loading && !result && !error && (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '12px',
              color: '#484F58',
              textAlign: 'center'
            }}>
              <p style={{ fontSize: '15px', color: '#8B949E' }}>
                Image ready. Select category and run detection.
              </p>
            </div>
          )}

          {loading && (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '20px'
            }}>
              <div style={{
                width: '48px', height: '48px', borderRadius: '50%',
                border: '2px solid #21262D',
                borderTop: '2px solid #58A6FF',
                animation: 'spin 0.8s linear infinite'
              }} />
              <div style={{ textAlign: 'center' }}>
                <p style={{ fontSize: '15px', marginBottom: '6px' }}>Running detection</p>
                <p style={{ fontSize: '13px', color: '#8B949E' }}>
                  Extracting DINOv2 features and scoring patches
                </p>
              </div>
              <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
            </div>
          )}

          {error && (
            <div style={{
              padding: '20px',
              borderRadius: '10px',
              background: 'rgba(248,81,73,0.05)',
              border: '1px solid rgba(248,81,73,0.2)'
            }}>
              <p style={{ fontSize: '14px', color: '#F85149', fontWeight: 500, marginBottom: '6px' }}>
                Detection failed
              </p>
              <p style={{ fontSize: '13px', color: '#8B949E' }}>{error}</p>
              <p style={{ fontSize: '12px', color: '#484F58', marginTop: '10px' }}>
                Make sure the backend is running: uvicorn api:app --reload --port 8000
              </p>
            </div>
          )}

          {result && !loading && (
            <>
              {/* Score */}
              <div>
                <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', marginBottom: '16px' }}>
                  <div>
                    <p style={{ fontSize: '12px', color: '#8B949E', marginBottom: '8px' }}>Anomaly score</p>
                    <p style={{
                      fontSize: '56px',
                      fontWeight: 700,
                      fontFamily: 'monospace',
                      lineHeight: 1,
                      color: isAnomaly ? '#F85149' : '#3FB950'
                    }}>
                      {result.score.toFixed(2)}
                    </p>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <p style={{ fontSize: '12px', color: '#8B949E', marginBottom: '4px' }}>Inference</p>
                    <p style={{ fontSize: '20px', fontFamily: 'monospace', color: '#E6EDF3' }}>
                      {result.inference_time_ms}ms
                    </p>
                  </div>
                </div>

                {/* Score bar */}
                <div style={{
                  height: '4px',
                  background: '#21262D',
                  borderRadius: '2px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${Math.min((result.score / 100) * 100, 100)}%`,
                    background: isAnomaly ? '#F85149' : '#3FB950',
                    borderRadius: '2px',
                    transition: 'width 0.8s ease-out'
                  }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px' }}>
                  <span style={{ fontSize: '11px', color: '#484F58', fontFamily: 'monospace' }}>0</span>
                  <span style={{ fontSize: '11px', color: '#484F58', fontFamily: 'monospace' }}>
                    threshold {THRESHOLD}
                  </span>
                  <span style={{ fontSize: '11px', color: '#484F58', fontFamily: 'monospace' }}>100</span>
                </div>
              </div>

              <div style={{ height: '1px', background: '#21262D' }} />

              {/* Visualization */}
              <div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: '16px'
                }}>
                  <p style={{ fontSize: '13px', color: '#8B949E' }}>Visualization</p>
                  <div style={{
                    display: 'flex',
                    background: '#161B22',
                    border: '1px solid #30363D',
                    borderRadius: '6px',
                    overflow: 'hidden'
                  }}>
                    {['original', 'heatmap', 'overlay'].map(v => (
                      <button
                        key={v}
                        onClick={() => setActiveView(v)}
                        style={{
                          padding: '6px 16px',
                          background: activeView === v ? '#58A6FF' : 'transparent',
                          color: activeView === v ? '#0D1117' : '#8B949E',
                          border: 'none',
                          fontSize: '13px',
                          fontWeight: activeView === v ? 600 : 400,
                          cursor: 'pointer',
                          fontFamily: 'Inter, sans-serif',
                          transition: 'all 0.15s',
                          textTransform: 'capitalize'
                        }}
                      >
                        {v}
                      </button>
                    ))}
                  </div>
                </div>

                <div style={{
                  borderRadius: '10px',
                  overflow: 'hidden',
                  background: '#161B22',
                  border: '1px solid #21262D',
                  position: 'relative',
                  aspectRatio: '1'
                }}>
                  {['original', 'heatmap', 'overlay'].map(v => (
                    <img
                      key={v}
                      src={views[v]}
                      alt={v}
                      style={{
                        position: 'absolute',
                        inset: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        opacity: activeView === v ? 1 : 0,
                        transition: 'opacity 0.3s ease'
                      }}
                    />
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
