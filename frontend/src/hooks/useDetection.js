import { useState, useCallback } from 'react'
import { detectAnomaly } from '../utils/api'

export function useDetection() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const detect = useCallback(async (imageFile, category) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await detectAnomaly(imageFile, category)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
    setLoading(false)
  }, [])

  return { result, loading, error, detect, reset }
}