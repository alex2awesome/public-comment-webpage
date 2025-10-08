import defaultTheme from 'tailwindcss/defaultTheme'

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f2f7ff',
          100: '#e6eeff',
          200: '#c5d8ff',
          300: '#a4c2ff',
          400: '#6d9eff',
          500: '#377aff',
          600: '#1f5ddb',
          700: '#1847a8',
          800: '#123175',
          900: '#0b1d47',
        },
        accent: {
          100: '#fef3c7',
          200: '#fde68a',
          500: '#f59e0b',
        },
      },
      fontFamily: {
        sans: ['Inter', ...defaultTheme.fontFamily.sans],
        display: ['"Work Sans"', ...defaultTheme.fontFamily.sans],
      },
      boxShadow: {
        card: '0 8px 24px rgba(15, 23, 42, 0.08)',
      },
    },
  },
  plugins: [],
}
