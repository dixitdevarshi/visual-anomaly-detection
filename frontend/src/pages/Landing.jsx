export default function Landing({ onStart }) {
  const stats = [
    { value: '0.9781', label: 'Mean AUROC' },
    { value: '15', label: 'Product Categories' },
    { value: '1.0', label: 'Perfect Score on 4 Categories' },
    { value: '167ms', label: 'Avg Inference Time' },
  ]

  const categories = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
  ]

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      <nav style={{
        padding: '20px 48px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid #21262D'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '28px', height: '28px', borderRadius: '6px',
            background: 'linear-gradient(135deg, #58A6FF, #1F6FEB)',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
          </div>
          <span style={{ fontWeight: 600, fontSize: '15px' }}>AnomalyVision</span>
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', color: '#8B949E', fontFamily: 'monospace' }}>
            DINOv2 + PatchCore
          </span>
          <div style={{ width: '1px', height: '16px', background: '#30363D' }} />
          <a
            href="https://github.com/dixitdevarshi"
            target="_blank"
            rel="noreferrer"
            style={{ color: '#8B949E', textDecoration: 'none', fontSize: '13px' }}
            >
            GitHub
          </a>
        </div>
      </nav>

      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '80px 48px 60px',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden'
      }}>

        <div style={{
          position: 'absolute',
          top: '10%',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '600px',
          height: '300px',
          background: 'radial-gradient(ellipse at center, rgba(88,166,255,0.08) 0%, transparent 70%)',
          pointerEvents: 'none'
        }} />

        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 14px',
          borderRadius: '20px',
          border: '1px solid #21262D',
          background: '#161B22',
          marginBottom: '32px',
          fontSize: '12px',
          color: '#8B949E'
        }}>
          <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#3FB950' }} />
          Unsupervised detection — no defect labels needed
        </div>

        <h1 style={{
          fontSize: '64px',
          fontWeight: 700,
          lineHeight: 1.1,
          marginBottom: '24px',
          letterSpacing: '-1.5px',
          maxWidth: '800px'
        }}>
          Find defects before
          <br />
          <span style={{ color: '#58A6FF' }}>they reach production</span>
        </h1>

        <p style={{
          fontSize: '18px',
          color: '#8B949E',
          maxWidth: '520px',
          lineHeight: 1.7,
          marginBottom: '48px'
        }}>
          Upload a product image. Get an anomaly score and a spatial heatmap
          showing exactly where the defect is, in under 200ms.
        </p>

        <div style={{ display: 'flex', gap: '12px', marginBottom: '80px' }}>
          <button
            onClick={onStart}
            style={{
              padding: '14px 32px',
              borderRadius: '8px',
              background: '#58A6FF',
              color: '#0D1117',
              border: 'none',
              fontSize: '15px',
              fontWeight: 600,
              cursor: 'pointer',
              fontFamily: 'Inter, sans-serif'
            }}
          >
            Try it now
          </button>
          <a
            href="https://github.com/dixitdevarshi/visual-anomaly-detection"
            target="_blank"
            rel="noreferrer"
            style={{
              padding: '14px 32px',
              borderRadius: '8px',
              background: 'transparent',
              color: '#E6EDF3',
              border: '1px solid #30363D',
              fontSize: '15px',
              fontWeight: 500,
              cursor: 'pointer',
              textDecoration: 'none',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              fontFamily: 'Inter, sans-serif'
            }}
          >
            View on GitHub
          </a>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '1px',
          background: '#21262D',
          borderRadius: '12px',
          overflow: 'hidden',
          width: '100%',
          maxWidth: '800px'
        }}>
          {stats.map(({ value, label }) => (
            <div key={label} style={{
              background: '#161B22',
              padding: '24px',
              textAlign: 'center'
            }}>
              <div style={{
                fontSize: '28px',
                fontWeight: 700,
                fontFamily: 'monospace',
                color: '#58A6FF',
                marginBottom: '4px'
              }}>{value}</div>
              <div style={{ fontSize: '12px', color: '#8B949E' }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ padding: '80px 48px', borderTop: '1px solid #21262D' }}>
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
          <p style={{ fontSize: '12px', color: '#8B949E', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '16px' }}>
            How it works
          </p>
          <h2 style={{ fontSize: '36px', fontWeight: 700, marginBottom: '48px', letterSpacing: '-0.5px' }}>
            Vision AI that learns what normal looks like
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
            {[
              {
                step: '01',
                title: 'Feature extraction',
                desc: 'DINOv2 ViT-B/14, frozen and pretrained on 142M images, splits each image into 256 patches and extracts a 768-dimensional feature vector per patch.'
              },
              {
                step: '02',
                title: 'Memory bank',
                desc: 'Patch features from defect-free training images are compressed into a memory bank via greedy coreset subsampling. No defect images required.'
              },
              {
                step: '03',
                title: 'Anomaly scoring',
                desc: 'At inference, each patch is compared against the memory bank. High distance from normal features means anomaly. Score is mapped to a heatmap.'
              }
            ].map(({ step, title, desc }) => (
              <div key={step} style={{
                padding: '28px',
                borderRadius: '10px',
                background: '#161B22',
                border: '1px solid #21262D'
              }}>
                <div style={{
                  fontFamily: 'monospace',
                  fontSize: '11px',
                  color: '#58A6FF',
                  marginBottom: '12px',
                  letterSpacing: '1px'
                }}>{step}</div>
                <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '10px' }}>{title}</h3>
                <p style={{ fontSize: '14px', color: '#8B949E', lineHeight: 1.7 }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ padding: '60px 48px', borderTop: '1px solid #21262D' }}>
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
          <p style={{ fontSize: '13px', color: '#8B949E', marginBottom: '20px' }}>
            Trained and evaluated on all 15 MVTec AD categories
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {categories.map(cat => (
              <span key={cat} style={{
                padding: '6px 14px',
                borderRadius: '6px',
                background: '#161B22',
                border: '1px solid #21262D',
                fontSize: '13px',
                color: '#8B949E',
                fontFamily: 'monospace'
              }}>
                {cat}
              </span>
            ))}
          </div>
        </div>
      </div>

      <footer style={{
        padding: '24px 48px',
        borderTop: '1px solid #21262D',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <span style={{ fontSize: '13px', color: '#8B949E' }}>
          Built by Devarshi Dixit · M.Sc. Intelligent Interactive Systems, Universität Bielefeld
        </span>
        <span style={{ fontSize: '12px', color: '#484F58', fontFamily: 'monospace' }}>
          CVPR 2022 · arXiv 2023
        </span>
      </footer>
    </div>
  )
}