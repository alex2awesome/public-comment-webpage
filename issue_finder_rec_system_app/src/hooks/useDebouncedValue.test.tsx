import { act, render, screen } from '@testing-library/react'
import { describe, expect, test, vi, beforeEach, afterEach } from 'vitest'
import { useDebouncedValue } from './useDebouncedValue'

const TestComponent = ({ value }: { value: string }) => {
  const debounced = useDebouncedValue(value, 100)
  return <span data-testid="value">{debounced}</span>
}

describe('useDebouncedValue', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  test('delays updates until the timeout elapses', () => {
    const { rerender } = render(<TestComponent value="initial" />)
    expect(screen.getByTestId('value')).toHaveTextContent('initial')

    rerender(<TestComponent value="updated" />)
    expect(screen.getByTestId('value')).toHaveTextContent('initial')

    act(() => {
      vi.advanceTimersByTime(99)
    })
    expect(screen.getByTestId('value')).toHaveTextContent('initial')

    act(() => {
      vi.advanceTimersByTime(1)
    })
    expect(screen.getByTestId('value')).toHaveTextContent('updated')
  })
})
