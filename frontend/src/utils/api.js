const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function detectAnomaly(imageFile, category) {
  const formData = new FormData()
  formData.append('file', imageFile)
  formData.append('category', category)

  const response = await fetch(`${API_BASE}/detect`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Detection failed')
  }

  return response.json()
}

export async function getCategories() {
  const response = await fetch(`${API_BASE}/categories`)
  if (!response.ok) throw new Error('Failed to fetch categories')
  return response.json()
}

export async function healthCheck() {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) throw new Error('API not reachable')
  return response.json()
}
